import json
import os
from utils import language_utils
import functools
print = functools.partial(print, flush=True)

class PlanDatasetKarpathy:
    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(self, images_root, annotation_json, verbose=True):
        self.images_path = images_root
        self.karpathy_train_list = []
        self.karpathy_val_list = []
        self.karpathy_test_list = []

        with open(annotation_json, 'r') as f:
            data = json.load(f)["images"]

        for idx, item in enumerate(data):
            entry = {}
            # automatically detect subfolder path
            entry["img_path"] = os.path.join(images_root, item['filepath'], item['filename'])
            entry["captions"] = [s["raw"] for s in item["sentences"]]
            entry["img_id"] = idx  # assign unique id if cocoid not present
            split = item.get("split", "train")
            if split == "train":
                self.karpathy_train_list.append(entry)
            elif split == "val":
                self.karpathy_val_list.append(entry)
            else:
                self.karpathy_test_list.append(entry)

        self.train_num_images = len(self.karpathy_train_list)
        self.val_num_images = len(self.karpathy_val_list)
        self.test_num_images = len(self.karpathy_test_list)

        if verbose:
            print(f"Num train: {self.train_num_images}, val: {self.val_num_images}, test: {self.test_num_images}")

        self.build_vocab(verbose)

    def build_vocab(self, verbose=True):
        tokenized_captions_list = []
        for img in self.karpathy_train_list:
            for caption in img["captions"]:
                tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
                tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
                tmp = language_utils.remove_punctuations(tmp)
                tokenized_captions_list.append(["SOS"] + language_utils.tokenize(tmp)[0] + ["EOS"])

        counter = {}
        for caption in tokenized_captions_list:
            for w in caption:
                counter[w] = counter.get(w, 0) + 1

        min_occurrences = 1
        vocab = ["PAD", "SOS", "EOS", "UNK"]
        for w, c in counter.items():
            if c >= min_occurrences and w not in vocab:
                vocab.append(w)

        self.num_caption_vocab = len(vocab)
        self.caption_word2idx_dict = {w: i for i, w in enumerate(vocab)}
        self.caption_idx2word_list = vocab
        self.max_seq_len = max(len(caption) for caption in tokenized_captions_list)
        if verbose:
            print(f"Vocab size: {self.num_caption_vocab}, max_seq_len: {self.max_seq_len}")

    def get_all_images_captions(self, dataset_split):
        if dataset_split == self.TestSet_ID:
            dataset = self.karpathy_test_list
        elif dataset_split == self.ValidationSet_ID:
            dataset = self.karpathy_val_list
        else:
            dataset = self.karpathy_train_list
        return [img["captions"] for img in dataset]

    def get_pad_token_str(self):
        return "PAD"

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict["UNK"]

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict["PAD"]

    def get_sos_token_str(self):
        return "SOS"

    def get_eos_token_str(self):
        return "EOS"
