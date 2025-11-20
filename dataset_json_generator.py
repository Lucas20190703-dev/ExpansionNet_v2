#!/usr/bin/env python3
"""
dataset_coco_generator_expansionnet.py

Generates dataset_coco.json for ExpansionNet_v2 using language_utils.py for tokenization.
Vocabulary is built from all tokenized captions.
"""

import os
import json
from utils.language_utils import tokenize  # use ExpansionNet tokenization

# -----------------------------
# CONFIGURATION
# -----------------------------
MATERIAL_FOLDER = 'github_ignore_material/raw_data/'
COCO_DATA_FOLDER = MATERIAL_FOLDER + 'MS_COCO_2014/'
TRAIN_CAPTIONS_FILE = COCO_DATA_FOLDER + 'annotations/captions_train2014.json'
VAL_CAPTIONS_FILE = COCO_DATA_FOLDER + 'annotations/captions_val2014.json'
OUTPUT_FILE = MATERIAL_FOLDER + 'dataset_coco.json'
TRAIN_FOLDER = COCO_DATA_FOLDER + 'train2014'
VAL_FOLDER = COCO_DATA_FOLDER + 'val2014'
VAL_SPLIT_RATIO = 0.5  # fraction of val images assigned to 'val' vs 'test'

# -----------------------------
# LOAD COCO CAPTIONS
# -----------------------------
with open(TRAIN_CAPTIONS_FILE, 'r') as f:
    train_data = json.load(f)

with open(VAL_CAPTIONS_FILE, 'r') as f:
    val_data = json.load(f)

# -----------------------------
# HELPER FUNCTION: BUILD IMAGE MAP
# -----------------------------
def build_image_map(coco_data, folder, split_name='train'):
    img_map = {}
    for idx, img in enumerate(coco_data['images']):
        img_map[img['id']] = {
            'filepath': folder,
            'filename': img['file_name'],
            'imgid': idx,
            'split': split_name,
            'sentids': [],
            'sentences': [],
            'cocoid': img['id']
        }
    return img_map

# -----------------------------
# BUILD TRAIN IMAGE MAP
# -----------------------------
train_img_map = build_image_map(train_data, TRAIN_FOLDER, 'train')

# -----------------------------
# BUILD VAL/TEST IMAGE MAP
# -----------------------------
val_img_map = {}
num_val_images = len(val_data['images'])
num_val_split = int(num_val_images * VAL_SPLIT_RATIO)

for idx, img in enumerate(val_data['images']):
    split = 'val' if idx < num_val_split else 'test'
    val_img_map[img['id']] = {
        'filepath': VAL_FOLDER,
        'filename': img['file_name'],
        'imgid': len(train_img_map) + idx,
        'split': split,
        'sentids': [],
        'sentences': [],
        'cocoid': img['id']
    }

# restval = val images not in val split
restval_img_map = {}
for img_id, img_entry in val_img_map.items():
    if img_entry['split'] != 'val':
        img_copy = img_entry.copy()
        img_copy['split'] = 'restval'
        restval_img_map[img_id] = img_copy

# -----------------------------
# COMBINE ALL IMAGE MAPS
# -----------------------------
all_img_map = {**train_img_map, **val_img_map, **restval_img_map}

# -----------------------------
# PROCESS CAPTIONS USING language_utils.py
# -----------------------------
all_tokens = set()

for ann in train_data['annotations'] + val_data['annotations']:
    img_entry = all_img_map[ann['image_id']]
    
    # Use ExpansionNet_v2 tokenization
    tokens = tokenize([ann['caption']])[0]
    all_tokens.update(tokens)
    
    sentence_entry = {
        'tokens': tokens,
        'raw': ann['caption'],
        'imgid': img_entry['imgid'],
        'sentid': ann['id']
    }
    img_entry['sentences'].append(sentence_entry)
    img_entry['sentids'].append(ann['id'])

# -----------------------------
# BUILD VOCABULARY MANUALLY
# -----------------------------
vocab = {
    'word2idx': {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3},
    'idx2word': {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
}

idx = 4
for token in sorted(all_tokens):
    vocab['word2idx'][token] = idx
    vocab['idx2word'][idx] = token
    idx += 1

# -----------------------------
# BUILD FINAL DATASET
# -----------------------------
dataset = {
    'images': list(all_img_map.values()),
    'vocab': vocab
}

# -----------------------------
# SAVE TO JSON
# -----------------------------
with open(OUTPUT_FILE, 'w') as f:
    json.dump(dataset, f)

# -----------------------------
# PRINT SUMMARY
# -----------------------------
print(f"dataset_coco.json generated!")
print(f"Total images: {len(dataset['images'])}")
print(f"Vocabulary size: {len(dataset['vocab']['word2idx'])}")

split_counts = {}
for img in dataset['images']:
    split_counts[img['split']] = split_counts.get(img['split'], 0) + 1
print(f"Split counts: {split_counts}")
