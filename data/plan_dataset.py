import json
import os
from utils import language_utils

class PlanDatasetKarpathy:
    TrainSet_ID = 0
    ValidationSet_ID = 1
    TestSet_ID = 2

    def __init__(self, images_root, annotation_json, verbose=True):
        self.images_root = images_root
        self.annotation_json = annotation_json
        self.verbose = verbose

        # Load annotation JSON
        with open(annotation_json, 'r', encoding='utf-8') as f:
            self.dataset_json = json.load(f)["images"]

        # Split into train/val/test lists
        self.karpathy_train_list = []
        self.karpathy_val_list = []
        self.karpathy_test_list = []

        self.caption_word2idx_dict = {}
        self.caption_idx2word_list = []

        self.max_seq_len = 0

        self._prepare_dataset()
        if verbose:
            print(f"Dataset loaded: {len(self.karpathy_train_list)} train, "
                  f"{len(self.karpathy_val_list)} val, {len(self.karpathy_test_list)} test samples.")

    def _prepare_dataset(self):
        """
        Populate train/val/test lists, build vocab
        Expected JSON format:
        [
            {"img_path": "image1.jpg", "captions": ["caption1", "caption2"], "split": "train"},
            {"img_path": "image2.jpg", "captions": ["caption1"], "split": "val"},
            ...
        ]
        """
        vocab_set = set()
        for item in self.dataset_json:
            # Resolve full image path
            img_path = os.path.join(self.images_root, item["img_path"])
            captions = item["captions"]
            # Track max sequence length
            for cap in captions:
                tokens = language_utils.tokenize([cap])[0]
                self.max_seq_len = max(self.max_seq_len, len(tokens) + 2)  # add SOS/EOS

            record = {"img_path": img_path, "captions": captions, "img_id": item.get("img_id", item["img_path"])}
            split = item.get("split", "train").lower()
            if split == "train":
                self.karpathy_train_list.append(record)
            elif split == "val":
                self.karpathy_val_list.append(record)
            elif split == "test":
                self.karpathy_test_list.append(record)

            for cap in captions:
                vocab_set.update(language_utils.tokenize([cap])[0])

        # Build word2idx and idx2word
        self.caption_word2idx_dict = {w: i + 4 for i, w in enumerate(sorted(vocab_set))}
        self.caption_word2idx_dict["PAD"] = 0
        self.caption_word2idx_dict["SOS"] = 1
        self.caption_word2idx_dict["EOS"] = 2
        self.caption_word2idx_dict["UNK"] = 3

        self.caption_idx2word_list = [None] * len(self.caption_word2idx_dict)
        for w, i in self.caption_word2idx_dict.items():
            self.caption_idx2word_list[i] = w

    def get_sos_token_str(self):
        return "SOS"

    def get_eos_token_str(self):
        return "EOS"

    def get_unk_token_str(self):
        return "UNK"

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict["PAD"]
    
    def get_unk_token_idx(self):
        return self.caption_word2idx_dict["UNK"]

    def get_all_images_captions(self, split_id):
        if split_id == self.TrainSet_ID:
            return [item["captions"] for item in self.karpathy_train_list]
        elif split_id == self.ValidationSet_ID:
            return [item["captions"] for item in self.karpathy_val_list]
        else:
            return [item["captions"] for item in self.karpathy_test_list]
