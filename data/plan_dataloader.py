import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.plan_dataset import PlanDatasetKarpathy
from utils import language_utils

# -----------------------------
# PyTorch Dataset wrapper
# -----------------------------
class PlanDatasetTorch(Dataset):
    """
    Wrapper around PlanDatasetKarpathy to make it a proper PyTorch Dataset
    """
    def __init__(self, plan_dataset, split_id):
        self.plan_dataset = plan_dataset
        self.split_id = split_id

        if split_id == PlanDatasetKarpathy.TrainSet_ID:
            self.data = plan_dataset.karpathy_train_list
        elif split_id == PlanDatasetKarpathy.ValidationSet_ID:
            self.data = plan_dataset.karpathy_val_list
        else:
            self.data = plan_dataset.karpathy_test_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Convert captions to token ids
        captions_tokenized = [
            [self.plan_dataset.caption_word2idx_dict.get(tok, self.plan_dataset.get_unk_token_idx())
             for tok in language_utils.tokenize([cap])[0]]
            for cap in item["captions"]
        ]
        return item["img_path"], captions_tokenized, item["img_id"]


# -----------------------------
# Custom collate function
# -----------------------------
def plan_collate_fn(batch, device="cuda", dataloader_mode="caption_wise"):
    """
    batch: list of tuples from PlanDatasetTorch: (img_path, captions_tokenized, img_id)
    """
    img_paths, captions_tokenized_list, img_ids = zip(*batch)
    
    batch_size = len(img_paths)
    
    # Dummy image tensor (replace with real preprocessing)
    batch_x = torch.randn(batch_size, 3, 384, 384, device=device)
    
    if dataloader_mode == "caption_wise":
        # For XE training, pick first caption per image
        batch_y_list = []
        for captions in captions_tokenized_list:
            cap_ids = torch.tensor(captions[0], dtype=torch.long)
            batch_y_list.append(cap_ids)
        # Pad captions to max length in batch
        batch_y = pad_sequence(batch_y_list, batch_first=True, padding_value=0).to(device)
        return batch_x, batch_y
    else:
        # For RL / SCST, return all tokenized captions
        return batch_x, captions_tokenized_list, img_ids


# -----------------------------
# DataLoader wrapper
# -----------------------------
class PlanDataLoader:
    """
    Windows-friendly PlanDataLoader for single/multi-GPU using DataParallel
    """
    def __init__(self, plan_dataset, batch_size=32, split_id=PlanDatasetKarpathy.TrainSet_ID,
                 resize_image_size=384, dataloader_mode="caption_wise", verbose=True):
        self.plan_dataset = plan_dataset
        self.batch_size = batch_size
        self.split_id = split_id
        self.resize_image_size = resize_image_size
        self.dataloader_mode = dataloader_mode

        self.torch_dataset = PlanDatasetTorch(plan_dataset, split_id)
        self.loader = DataLoader(
            self.torch_dataset,
            batch_size=self.batch_size,
            shuffle=(split_id == PlanDatasetKarpathy.TrainSet_ID),
            num_workers=0,  # adjust based on CPU cores
            pin_memory=True,
            collate_fn=lambda batch: plan_collate_fn(batch, device="cuda", dataloader_mode=self.dataloader_mode)
        )
        self.iter_loader = iter(self.loader)

        if verbose:
            print(f"PlanDataLoader initialized with {len(self.torch_dataset)} samples, "
                  f"batch_size={batch_size}, mode={dataloader_mode}")

    def get_next_batch(self, device="cuda", get_also_image_idxes=False):
        try:
            batch = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            batch = next(self.iter_loader)

        if get_also_image_idxes and self.dataloader_mode != "caption_wise":
            return batch
        else:
            return batch  # batch_x, batch_y for XE or batch_x, captions for RL

    def get_num_batches(self):
        return len(self.loader)
