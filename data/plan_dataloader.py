import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as T
from data.plan_dataset import PlanDatasetKarpathy
from utils import language_utils

# -----------------------------
# PyTorch Dataset wrapper
# -----------------------------
class PlanDatasetTorch(Dataset):
    """Wrap PlanDatasetKarpathy for PyTorch."""
    def __init__(self, plan_dataset, split_id, use_images=True, resize_image_size=384):
        self.plan_dataset = plan_dataset
        self.split_id = split_id
        self.use_images = use_images
        self.resize_image_size = resize_image_size

        if split_id == PlanDatasetKarpathy.TrainSet_ID:
            self.data = plan_dataset.karpathy_train_list
        elif split_id == PlanDatasetKarpathy.ValidationSet_ID:
            self.data = plan_dataset.karpathy_val_list
        else:
            self.data = plan_dataset.karpathy_test_list

        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((resize_image_size, resize_image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image or feature
        if self.use_images:
            img_path = item["img_path"]
            image = Image.open(img_path).convert("RGB")
            image = self.transforms(image)
            enc_x_num_pads = None  # End-to-End requires no padding
        else:
            # Precomputed features
            img_id = item["img_id"]
            image = torch.tensor(self.plan_dataset.hdf5_file[f'{img_id}_features'][()])
            enc_x_num_pads = [0] * len(image)  # for feature padding logic

        # Convert captions to token ids
        captions_tokenized = [
            [self.plan_dataset.caption_word2idx_dict.get(tok, self.plan_dataset.get_unk_token_idx())
             for tok in language_utils.tokenize([cap])[0]]
            for cap in item["captions"]
        ]

        return image, captions_tokenized, item["img_id"], enc_x_num_pads


# -----------------------------
# Collate function
# -----------------------------
def plan_collate_fn(batch, device="cuda", dataloader_mode="caption_wise"):
    images, captions_list, img_ids, enc_num_pads_list = zip(*batch)
    batch_x = torch.stack(images).to(device)

    if dataloader_mode == "caption_wise":
        batch_y_list = []
        for captions in captions_list:
            # Pick first caption per image for XE
            batch_y_list.append(torch.tensor(captions[0], dtype=torch.long))
        batch_y = pad_sequence(batch_y_list, batch_first=True, padding_value=0).to(device)
        enc_x_num_pads = None if enc_num_pads_list[0] is None else torch.tensor(enc_num_pads_list).to(device)
        return batch_x, batch_y, enc_x_num_pads
    else:
        # RL / SCST mode, return all captions
        enc_x_num_pads = None if enc_num_pads_list[0] is None else torch.tensor(enc_num_pads_list).to(device)
        return batch_x, captions_list, img_ids, enc_x_num_pads


# -----------------------------
# DataLoader wrapper
# -----------------------------
class PlanDataLoader:
    """Iterable Windows-compatible DataLoader for PlanDataset."""

    def __init__(self, plan_dataset, batch_size=32, split_id=PlanDatasetKarpathy.TrainSet_ID,
                 resize_image_size=384, dataloader_mode="caption_wise",
                 use_images=True, num_workers=0, device="cuda", verbose=True):

        self.plan_dataset = plan_dataset
        self.batch_size = batch_size
        self.split_id = split_id
        self.device = device
        self.dataloader_mode = dataloader_mode
        self.use_images = use_images

        self.torch_dataset = PlanDatasetTorch(plan_dataset, split_id, use_images, resize_image_size)
        self.loader = DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
            shuffle=(split_id == PlanDatasetKarpathy.TrainSet_ID),
            num_workers=num_workers,
            pin_memory=False, #(device=="cuda"),
            collate_fn=lambda batch: plan_collate_fn(batch, device=self.device, dataloader_mode=self.dataloader_mode)
        )

        if verbose:
            print(f"[PlanDataLoader] Initialized: {len(self.torch_dataset)} samples, "
                  f"batch_size={self.batch_size}, mode={self.dataloader_mode}, use_images={self.use_images}")

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def get_next_batch(self):
        # If you want a method like your old API
        if not hasattr(self, "_iter"):
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)
