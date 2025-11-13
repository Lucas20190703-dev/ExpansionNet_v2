import argparse
import random
import torch
import torch.optim as optim
from torch.nn import DataParallel
from time import time
import functools

from data.plan_dataset import PlanDatasetKarpathy
from data.plan_dataloader import PlanDataLoader
from losses.loss import LabelSmoothingLoss
from losses.reward import ReinforceCiderReward
from optims.radam import RAdam
from utils import language_utils

print = functools.partial(print, flush=True)


# ---------------------------
# Training loop
# ---------------------------
def train(train_args, plan_dataset, data_loader, model, optimizer, sched, max_len):
    if not train_args.reinforce:
        loss_function = LabelSmoothingLoss(smoothing_coeff=0.1)
        loss_function.to('cuda')
    else:
        num_sampled_captions = 5
        training_references = plan_dataset.get_all_images_captions(PlanDatasetKarpathy.TrainSet_ID)
        reinforce_reward = ReinforceCiderReward(
            training_references,
            plan_dataset.get_eos_token_str(),
            num_sampled_captions,
            device='cuda'
        )

    total_iter = data_loader.get_num_batches() * train_args.num_epochs
    print(f"Training for {train_args.num_epochs} epochs, {total_iter} iterations")

    for it in range(total_iter):
        if train_args.reinforce:
            batch_x, batch_captions, image_indices = data_loader.get_next_batch(get_also_image_idxes=True)
        else:
            batch_x, batch_y = data_loader.get_next_batch(device="cuda")

        if batch_x is None:
            data_loader.set_epoch_it(data_loader.epoch_it + 1, verbose=True)
            continue

        batch_x = batch_x.cuda()
        model.train()

        if not train_args.reinforce:
            batch_y = batch_y.cuda()
            pred_logprobs = model(enc_x=batch_x,
                                  dec_x=batch_y[:, :-1],
                                  apply_softmax=False)
            loss = loss_function(pred_logprobs, batch_y[:, 1:], plan_dataset.get_pad_token_idx())
        else:
            pred_logprobs = model(enc_x=batch_x,
                                  dec_x=None,
                                  apply_softmax=True)
            loss = reinforce_reward(pred_logprobs, batch_captions, image_indices)

        loss.backward()

        if (it + 1) % train_args.num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            sched.step()

        if (it + 1) % train_args.print_every_iter == 0:
            print(f"Iter {it + 1}: loss={loss.item():.4f}")


# ---------------------------
# Main training setup
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ExpansionNet_v2 on Plan dataset")
    parser.add_argument("--images_path", type=str, default="./github_ignore_material/plan_dataset/images/")
    parser.add_argument("--annotations_path", type=str, default="./github_ignore_material/plan_dataset/annotations/dataset_plan.json")
    parser.add_argument("--save_path", type=str, default="./checkpoints_plan/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--reinforce", type=bool, default=False)
    parser.add_argument("--is_end_to_end", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_accum", type=int, default=1)
    parser.add_argument("--print_every_iter", type=int, default=100)
    args = parser.parse_args()

    # Use Namespace for drop_args and model_args
    drop_args = argparse.Namespace(enc=0.1, dec=0.1, enc_input=0.1, dec_input=0.1, other=0.1)
    model_args = argparse.Namespace(model_dim=512, N_enc=3, N_dec=3, drop_args=drop_args)
    optim_args = argparse.Namespace(lr=args.lr)

    # Load dataset
    plan_dataset = PlanDatasetKarpathy(images_root=args.images_path,
                                       annotation_json=args.annotations_path,
                                       verbose=True)

    # Initialize dataloader
    data_loader = PlanDataLoader(plan_dataset=plan_dataset,
                                 batch_size=args.batch_size,
                                 split_id=PlanDatasetKarpathy.TrainSet_ID,
                                 verbose=True)

    # Initialize model
    if args.is_end_to_end:
        from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
        model = End_ExpansionNet_v2(
            swin_img_size=384, swin_patch_size=4, swin_in_chans=3,
            swin_embed_dim=192, swin_depths=[2,2,18,2], swin_num_heads=[6,12,24,48],
            swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True,
            swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.1,
            swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
            swin_qk_scale=None,
            swin_use_checkpoint=False,
            final_swin_dim=1536,
            d_model=model_args.model_dim, N_enc=model_args.N_enc, N_dec=model_args.N_dec,
            num_heads=8, ff=2048, num_exp_enc_list=[32,64,128,256,512], num_exp_dec=16,
            output_word2idx=plan_dataset.caption_word2idx_dict,
            output_idx2word=plan_dataset.caption_idx2word_list,
            max_seq_len=plan_dataset.max_seq_len + 20,
            drop_args=model_args.drop_args,
        )
    else:
        from models.ExpansionNet_v2 import ExpansionNet_v2
        model = ExpansionNet_v2(
            d_model=model_args.model_dim, N_enc=model_args.N_enc, N_dec=model_args.N_dec,
            num_heads=8, ff=2048, num_exp_enc_list=[32,64,128,256,512], num_exp_dec=16,
            output_word2idx=plan_dataset.caption_word2idx_dict,
            output_idx2word=plan_dataset.caption_idx2word_list,
            max_seq_len=plan_dataset.max_seq_len + 20,
            drop_args=model_args.drop_args,
            img_feature_dim=1536
        )

    # Use GPU
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # Optimizer
    optimizer = RAdam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: 1.0)

    # Start training
    train(args, plan_dataset, data_loader, model, optimizer, sched, plan_dataset.max_seq_len + 20)


if __name__ == "__main__":
    main()
