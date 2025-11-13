import os
import random
import argparse
from argparse import Namespace

import torch
from torch import nn
from torch.optim import Adam

from data.plan_dataset import PlanDatasetKarpathy
from data.plan_dataloader import PlanDataLoader

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils import language_utils
from losses.loss import LabelSmoothingLoss

import functools
print = functools.partial(print, flush=True)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train(model, dataloader, optimizer, loss_fn, device, num_epochs=10, print_every=50):
    model.train()
    total_iters = len(dataloader) * num_epochs
    print(f"Training for {num_epochs} epochs, {total_iters} iterations")

    iter_count = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            iter_count += 1
            batch_images, batch_captions, enc_x_num_pads = batch
            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)

            optimizer.zero_grad()
            pred_logprobs = model(enc_x=batch_images, enc_x_num_pads=None, dec_x=batch_captions[:, :-1], apply_softmax=False)
            loss = loss_fn(pred_logprobs, batch_captions[:, 1:], pad_idx=0)
            loss.backward()
            optimizer.step()

            if iter_count % print_every == 0:
                print(f"Epoch {epoch+1} Iter {iter_count}/{total_iters} - Loss: {loss.item():.4f}")

    print("Training finished.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train End_ExpansionNet_v2 on Plan dataset (Windows)")
    parser.add_argument("--images_path", type=str, default="./github_ignore_material/plan_dataset/images/")
    parser.add_argument("--annotations_path", type=str, default="./github_ignore_material/plan_dataset/annotations/dataset_plan.json")
    parser.add_argument("--save_path", type=str, default="./checkpoints_plan/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--print_every", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Plan dataset
    plan_dataset = PlanDatasetKarpathy(
        images_root=args.images_path,
        annotation_json=args.annotations_path,
        verbose=True
    )

    # Windows-compatible DataLoader
    dataloader = PlanDataLoader(
        plan_dataset,
        batch_size=args.batch_size,
        split_id=PlanDatasetKarpathy.TrainSet_ID,
        resize_image_size=384,
        dataloader_mode="caption_wise",  # XE training
        use_images=True,
        device=device,
        verbose=True
    )

    # Initialize End_ExpansionNet_v2
    img_size = 384
    model = End_ExpansionNet_v2(
        swin_img_size=img_size,
        swin_patch_size=4,
        swin_in_chans=3,
        swin_embed_dim=192,
        swin_depths=[2, 2, 18, 2],
        swin_num_heads=[6, 12, 24, 48],
        swin_window_size=12,
        swin_mlp_ratio=4.,
        swin_qkv_bias=True,
        swin_drop_rate=0.0,
        swin_attn_drop_rate=0.0,
        swin_drop_path_rate=0.1,
        swin_norm_layer=nn.LayerNorm,
        swin_ape=False,
        swin_patch_norm=True,
        swin_qk_scale=None,
        swin_use_checkpoint=False,
        final_swin_dim=1536,
        d_model=512,
        N_enc=3,
        N_dec=3,
        num_heads=8,
        ff=2048,
        num_exp_enc_list=[32, 64, 128, 256, 512],
        num_exp_dec=16,
        output_word2idx=plan_dataset.caption_word2idx_dict,
        output_idx2word=plan_dataset.caption_idx2word_list,
        max_seq_len=plan_dataset.max_seq_len + 20,
        drop_args=Namespace(enc=0.1, dec=0.1, enc_input=0.1, dec_input=0.1, other=0.1),
        rank=0
    ).to(device)

    # Loss and optimizer
    loss_fn = LabelSmoothingLoss(smoothing_coeff=0.1, rank=0).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train
    train(model, dataloader, optimizer, loss_fn, device,
          num_epochs=args.num_epochs, print_every=args.print_every)

    # Save model
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, "end_expansionnet_v2_plan.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")
