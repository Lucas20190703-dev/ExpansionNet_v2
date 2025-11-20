import os
import json
import random
from PIL import Image, ImageDraw

# =========================================
# CONFIG
# =========================================
OUT_DIR = "mini_coco"
TRAIN_IMAGES = 5
VAL_IMAGES = 3
TEST_IMAGES = 2
IMAGE_SIZE = (256, 256)

FAKE_CAPTIONS = [
    "A person standing in an empty room.",
    "A small dog is looking at the camera.",
    "A car driving down the street.",
    "A group of people sitting at a table.",
    "A cat sleeping on a chair.",
    "A cup on top of a wooden desk.",
    "A tree beside a quiet lake.",
    "A red square image used for testing caption models.",
    "A synthetic image with simple colors.",
    "A purple-toned test image for debugging caption pipelines."
]

# =========================================
# HELPERS
# =========================================

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_fake_image(path, idx):
    img = Image.new(
        "RGB",
        IMAGE_SIZE,
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    )
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"IMG {idx}", fill=(0,0,0))
    img.save(path)

def build_captions_json(image_infos, split):
    """
    Build MS-COCO format caption JSON.
    """
    annotations = []
    ann_id = 1
    
    for img in image_infos:
        # each image gets 5 fake captions
        for _ in range(5):
            caption_text = random.choice(FAKE_CAPTIONS)
            annotations.append({
                "id": ann_id,
                "image_id": img["id"],
                "caption": caption_text
            })
            ann_id += 1

    coco = {
        "info": {
            "description": f"Mini COCO {split} Dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": image_infos,
        "annotations": annotations
    }

    return coco

# =========================================
# GENERATE DATASET
# =========================================

print("Creating mini COCO caption dataset...")

# Create directories
ensure(OUT_DIR)
ensure(os.path.join(OUT_DIR, "annotations"))
ensure(os.path.join(OUT_DIR, "train2014"))
ensure(os.path.join(OUT_DIR, "val2014"))
ensure(os.path.join(OUT_DIR, "test2014"))

# --- TRAIN ---
train_images = []
for i in range(TRAIN_IMAGES):
    file_name = f"COCO_train2014_{i:012d}.jpg"
    path = os.path.join(OUT_DIR, "train2014", file_name)
    make_fake_image(path, i)

    train_images.append({
        "id": i,
        "file_name": file_name,
        "width": IMAGE_SIZE[0],
        "height": IMAGE_SIZE[1]
    })

# --- VAL ---
val_images = []
for i in range(VAL_IMAGES):
    file_name = f"COCO_val2014_{i:012d}.jpg"
    path = os.path.join(OUT_DIR, "val2014", file_name)
    make_fake_image(path, i + 10000)

    val_images.append({
        "id": i + 10000,
        "file_name": file_name,
        "width": IMAGE_SIZE[0],
        "height": IMAGE_SIZE[1]
    })

# --- TEST ---
test_images = []
for i in range(TEST_IMAGES):
    file_name = f"COCO_test2014_{i:012d}.jpg"
    path = os.path.join(OUT_DIR, "test2014", file_name)
    make_fake_image(path, i + 20000)

    test_images.append({
        "id": i + 20000,
        "file_name": file_name,
        "width": IMAGE_SIZE[0],
        "height": IMAGE_SIZE[1]
    })

# --- JSON FILES ---
train_json = build_captions_json(train_images, "train2014")
val_json = build_captions_json(val_images, "val2014")
test_json = build_captions_json(test_images, "test2014")

with open(os.path.join(OUT_DIR, "annotations", "captions_train2014.json"), "w", encoding="utf-8") as f:
    json.dump(train_json, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "annotations", "captions_val2014.json"), "w", encoding="utf-8") as f:
    json.dump(val_json, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUT_DIR, "annotations", "captions_test2014.json"), "w", encoding="utf-8") as f:
    json.dump(test_json, f, indent=2, ensure_ascii=False)

print("Mini COCO caption dataset created successfully!")
print(f"Directory: {OUT_DIR}")
