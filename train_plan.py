import argparse
import os
import random
from argparse import Namespace
from time import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from data.plan_dataset import PlanDatasetKarpathy
from data.plan_dataloader import PlanDataLoader
from test import compute_evaluation_loss, evaluate_model_on_set
from losses.loss import LabelSmoothingLoss
from losses.reward import ReinforceCiderReward
from optims.radam import RAdam
from utils import language_utils
from utils.args_utils import str2bool
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint, partially_load_state_dict

import functools
print = functools.partial(print, flush=True)

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ------------------- Training Loop -------------------
def train(rank, train_args, path_args, ddp_model, plan_dataset, data_loader,
          optimizer, sched, max_len):

    if not train_args.reinforce:
        loss_function = LabelSmoothingLoss(smoothing_coeff=0.1, rank=rank).to(rank)
    else:
        num_sampled_captions = 5
        training_references = plan_dataset.get_all_images_captions(PlanDatasetKarpathy.TrainSet_ID)
        reinforce_reward = ReinforceCiderReward(
            training_references,
            plan_dataset.get_eos_token_str(),
            num_sampled_captions,
            rank
        )

    total_iter = data_loader.get_num_batches() * train_args.num_epochs
    print(f"[GPU {rank}] Training for {train_args.num_epochs} epochs, {total_iter} iterations")

    for it in range(total_iter):
        # Get batch
        if train_args.reinforce:
            batch_x, batch_captions, image_indices = data_loader.get_next_batch(rank, get_also_image_idxes=True)
        else:
            batch_x, batch_y = data_loader.get_next_batch(rank)

        if batch_x is None:
            data_loader.set_epoch_it(data_loader.epoch_it + 1, verbose=True)
            continue

        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        batch_x = batch_x.to(device)

        ddp_model.train()
        if not train_args.reinforce:
            batch_y = batch_y.to(device)
            pred_logprobs = ddp_model(enc_x=batch_x, dec_x=batch_y[:, :-1], apply_softmax=False)
            loss = loss_function(pred_logprobs, batch_y[:, 1:], plan_dataset.get_pad_token_idx())
        else:
            pred_logprobs = ddp_model(enc_x=batch_x, dec_x=None, apply_softmax=True)
            loss = reinforce_reward(pred_logprobs, batch_captions, image_indices)

        loss.backward()

        if (it + 1) % train_args.num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            sched.step()

        if (it + 1) % train_args.print_every_iter == 0:
            print(f"[GPU {rank}] Iter {it + 1}: loss={loss.item():.4f}")

# ------------------- Distributed Training -------------------
def distributed_train(rank, world_size, model_args, optim_args,
                      plan_dataset, array_of_init_seeds, model_max_len,
                      train_args, path_args):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = train_args.ddp_sync_port

    # Use GLOO backend for Windows
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    img_size = 384
    if train_args.is_end_to_end:
        from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
        model = End_ExpansionNet_v2(
            swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
            swin_embed_dim=192, swin_depths=[2,2,18,2], swin_num_heads=[6,12,24,48],
            swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True,
            swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.1,
            swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
            final_swin_dim=1536,
            d_model=model_args.model_dim, N_enc=model_args.N_enc,
            N_dec=model_args.N_dec, num_heads=8, ff=2048,
            num_exp_enc_list=[32,64,128,256,512],
            num_exp_dec=16,
            output_word2idx=plan_dataset.caption_word2idx_dict,
            output_idx2word=plan_dataset.caption_idx2word_list,
            max_seq_len=model_max_len, drop_args=model_args.drop_args,
            rank=rank
        )
    else:
        from models.ExpansionNet_v2 import ExpansionNet_v2
        model = ExpansionNet_v2(
            d_model=model_args.model_dim, N_enc=model_args.N_enc,
            N_dec=model_args.N_dec, num_heads=8, ff=2048,
            num_exp_enc_list=[32,64,128,256,512],
            num_exp_dec=16,
            output_word2idx=plan_dataset.caption_word2idx_dict,
            output_idx2word=plan_dataset.caption_idx2word_list,
            max_seq_len=model_max_len, drop_args=model_args.drop_args,
            img_feature_dim=1536,
            rank=rank
        )

    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    dataloader_mode = 'image_wise' if train_args.reinforce else 'caption_wise'
    data_loader = PlanDataLoader(
        plan_dataset=plan_dataset,
        array_of_init_seeds=array_of_init_seeds,
        batch_size=train_args.batch_size,
        resize_image_size=img_size,
        dataloader_mode=dataloader_mode,
        rank=rank,
        num_procs=world_size,
        verbose=True
    )

    optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=optim_args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: 1.0)

    train(rank, train_args, path_args, ddp_model, plan_dataset, data_loader,
          optimizer, sched, model_max_len)

    print(f"[GPU {rank}] Training finished.")
    dist.destroy_process_group()

# ------------------- Spawn Processes -------------------
def spawn_train_processes(model_args, optim_args, plan_dataset, train_args, path_args):
    world_size = min(torch.cuda.device_count(), train_args.num_gpus)
    print(f"Using {world_size} GPU(s)")

    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs * 2)]
    mp.spawn(distributed_train,
             args=(world_size, model_args, optim_args, plan_dataset,
                   array_of_init_seeds, plan_dataset.max_seq_len + 20, train_args, path_args),
             nprocs=world_size,
             join=True)

# ------------------- Main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default="./github_ignore_material/plan_dataset/images/")
    parser.add_argument("--annotations_path", type=str, default="./github_ignore_material/plan_dataset/annotations/dataset_plan.json")
    parser.add_argument("--save_path", type=str, default="./checkpoints_plan/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ddp_sync_port", type=str, default="12355")
    parser.add_argument("--is_end_to_end", type=str2bool, default=True)
    parser.add_argument("--reinforce", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_accum", type=int, default=1)
    parser.add_argument("--print_every_iter", type=int, default=100)
    args = parser.parse_args()

    drop_args = Namespace(enc=0.1, dec=0.1, enc_input=0.1, dec_input=0.1, other=0.1)
    model_args = Namespace(model_dim=512, N_enc=3, N_dec=3, drop_args=drop_args)
    optim_args = Namespace(lr=args.lr)
    path_args = Namespace(save_path=args.save_path)
    train_args = Namespace(is_end_to_end=args.is_end_to_end,
                           batch_size=args.batch_size,
                           num_accum=args.num_accum,
                           num_gpus=args.num_gpus,
                           ddp_sync_port=args.ddp_sync_port,
                           reinforce=args.reinforce,
                           num_epochs=args.num_epochs,
                           print_every_iter=args.print_every_iter)

    plan_dataset = PlanDatasetKarpathy(
        images_root=args.images_path,
        annotation_json=args.annotations_path,
        verbose=True
    )

    spawn_train_processes(model_args, optim_args, plan_dataset, train_args, path_args)
