import random
import torch
from PIL import Image as PIL_Image
import torchvision
import h5py
from time import time
from utils import language_utils
from data.transparent_data_loader import TransparentDataLoader
import functools
print = functools.partial(print, flush=True)

class PlanDataLoader(TransparentDataLoader):

    def __init__(self, plan_dataset, array_of_init_seeds, batch_size,
                 rank=0, num_procs=1, dataloader_mode='caption_wise',
                 resize_image_size=384, verbose=False):
        super().__init__()
        self.plan_dataset = plan_dataset
        self.dataloader_mode = dataloader_mode
        self.rank = rank
        self.num_procs = num_procs
        self.batch_size = batch_size
        self.array_of_init_seeds = array_of_init_seeds * 10
        self.epoch_it = 0
        self.use_images_instead_of_features = True
        self.resize_image_size = resize_image_size

        preprocess_layers = [
            torchvision.transforms.Resize((resize_image_size, resize_image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                             std=[0.229,0.224,0.225])
        ]
        self.preprocess = torchvision.transforms.Compose(preprocess_layers)

        self.batch_it = [0] * num_procs
        self.image_idx_x = [[] for _ in range(num_procs)]
        self.caption_y = [[] for _ in range(num_procs)]

        self.set_epoch_it(0, verbose=verbose)

    def tokenize_caption(self, caption_text):
        tmp = language_utils.lowercase_and_clean_trailing_spaces([caption_text])
        tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
        tmp = language_utils.remove_punctuations(tmp)
        tokens = ["SOS"] + language_utils.tokenize(tmp)[0] + ["EOS"]
        out = [tok if tok in self.plan_dataset.caption_word2idx_dict else "UNK" for tok in tokens]
        return out

    def init_epoch(self, epoch_it, verbose=False):
        start_time = time()
        random.seed(self.array_of_init_seeds[epoch_it])
        self.batch_it = [0] * self.num_procs
        self.image_idx_x = [[] for _ in range(self.num_procs)]
        self.caption_y = [[] for _ in range(self.num_procs)]

        pairs = []
        for img_idx in range(self.plan_dataset.train_num_images):
            n_caps = len(self.plan_dataset.karpathy_train_list[img_idx]["captions"])
            for cap_idx in range(n_caps):
                pairs.append((img_idx, cap_idx))
        random.shuffle(pairs)
        tail = len(pairs) % (self.batch_size * self.num_procs)
        if tail:
            pairs = pairs[:-tail]

        proc_batches = [[] for _ in range(self.num_procs)]
        proc_caps = [[] for _ in range(self.num_procs)]
        i = 0
        while i < len(pairs):
            for p in range(self.num_procs):
                img_idx, cap_idx = pairs[i]
                proc_batches[p].append(img_idx)
                proc_caps[p].append(self.tokenize_caption(self.plan_dataset.karpathy_train_list[img_idx]["captions"][cap_idx]))
                i += 1
            if i % self.batch_size == 0:
                for p in range(self.num_procs):
                    self.image_idx_x[p].append(proc_batches[p])
                    self.caption_y[p].append(proc_caps[p])
                    proc_batches[p] = []
                    proc_caps[p] = []

        self.num_batches = len(self.image_idx_x[0])
        if verbose:
            print(f"[Rank {self.rank}] Init epoch {epoch_it} -> {self.num_batches} batches ({time() - start_time:.2f}s)")

    def pad_batch(self, captions, pad_token):
        max_len = max(len(c) for c in captions)
        padded = [c + [pad_token]*(max_len - len(c)) for c in captions]
        return padded

    def get_next_batch(self, idx_proc=0):
        if self.batch_it[idx_proc] >= self.num_batches:
            return None
        image_indices = self.image_idx_x[idx_proc][self.batch_it[idx_proc]]
        caption_sequences = self.caption_y[idx_proc][self.batch_it[idx_proc]]
        self.batch_it[idx_proc] += 1

        images = [self.preprocess(PIL_Image.open(self.plan_dataset.karpathy_train_list[idx]["img_path"]).convert("RGB")) for idx in image_indices]
        batch_images = torch.stack(images)

        pad_token = self.plan_dataset.get_pad_token_str()
        padded_captions = self.pad_batch(caption_sequences, pad_token)
        token_indices = [[self.plan_dataset.caption_word2idx_dict.get(tok, self.plan_dataset.get_unk_token_idx()) for tok in seq] for seq in padded_captions]
        token_tensor = torch.tensor(token_indices, dtype=torch.long)

        return batch_images, token_tensor

    def set_epoch_it(self, epoch, verbose=False):
        self.epoch_it = epoch
        self.init_epoch(epoch, verbose=verbose)

    def get_num_batches(self):
        return self.num_batches

    def get_epoch_it(self):
        return self.epoch_it
