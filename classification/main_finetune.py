# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import importlib
import warnings

from collections import defaultdict
from torchvision import datasets, transforms

import torch

# import torchinfo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


import timm

# assert timm.__version__ == "0.5.4"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_spikformer as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.kd_loss import DistillationLoss
from util.datasets import CIFAR10CDataset

from ppformer import *



from engine_finetune import train_one_epoch, evaluate
from timm.data import create_loader

import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: Define string-to-class mapping
transformer_dict = {
    "lidiff": LiDiffTransformer,
    "pp": PPTransformer,
    "spiking": SpikingTransformer,
    "token": TokenSpikingTransformer,
    "pptoken": PushPullTransformer
}

def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)

    # Basic training params
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU (effective batch size = batch_size * accum_iter * #GPUs)")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--accum_iter", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--finetune", default="", help="Finetune from checkpoint")
    parser.add_argument("--data_path", default="./data/cifar10/", type=str, help="Dataset path")
    parser.add_argument("--dataset", default="CIFAR10")

    # Model parameters
    parser.add_argument("--model", default="spikformer_8_384_CAFormer", type=str, metavar="MODEL", help="Model name")
    parser.add_argument("--model_mode", default="ms", type=str, help="Model mode")
    parser.add_argument("--input_size", default=224, type=int, help="Input image size")
    parser.add_argument("--drop_path", default=0.1, type=float, help="Drop path rate")

    # Optimizer
    parser.add_argument("--clip_grad", default=None, type=float, help="Gradient clipping (None = no clipping)")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay")
    parser.add_argument("--lr", default=None, type=float, help="Absolute learning rate")
    parser.add_argument("--blr", default=6e-4, type=float, help="Base LR: base_lr * total_batch_size / 256")
    parser.add_argument("--layer_decay", default=1.0, type=float, help="Layer-wise learning rate decay")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Min learning rate")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Warmup epochs")

    # Augmentation
    parser.add_argument("--color_jitter", default=None, type=float, help="Color jitter (if not using RandAug)")
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str, help="AutoAugment policy")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing")

    # Random Erase
    parser.add_argument("--reprob", default=0.25, type=float, help="Random erase probability")
    parser.add_argument("--remode", default="pixel", type=str, help="Random erase mode")
    parser.add_argument("--recount", default=1, type=int, help="Random erase count")
    parser.add_argument("--resplit", action="store_true", help="Don’t erase first augmentation split")

    # Mixup / Cutmix
    parser.add_argument("--mixup", default=0.0, type=float, help="Mixup alpha")
    parser.add_argument("--cutmix", default=0.0, type=float, help="Cutmix alpha")
    parser.add_argument("--cutmix_minmax", nargs="+", type=float, default=None, help="Cutmix min/max ratio")
    parser.add_argument("--mixup_prob", default=1.0, type=float, help="Probability of mixup/cutmix")
    parser.add_argument("--mixup_switch_prob", default=0.5, type=float, help="Probability to switch to cutmix")
    parser.add_argument("--mixup_mode", default="batch", type=str, help="Mixup mode: batch | pair | elem")

    # Finetuning
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="Use class token instead of global pooling")
    parser.add_argument("--time_steps", default=1, type=int)

    # Dataset
    parser.add_argument("--nb_classes", default=1000, type=int, help="Number of classes")
    parser.add_argument("--output_dir", default="/raid/ligq/htx/spikemae/output_dir", help="Output directory")
    parser.add_argument("--log_dir", default="/raid/ligq/htx/spikemae/output_dir", help="Tensorboard log directory")
    parser.add_argument("--c_eval", action='store_true')

    # System
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--dist_eval", action="store_true", default=False, help="Enable distributed evaluation")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin memory in DataLoader")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
    parser.set_defaults(pin_mem=True)

    # Distillation
    parser.add_argument("--kd", action="store_true", default=False, help="Enable knowledge distillation")
    parser.add_argument("--teacher_model", default="caformer_b36_in21ft1k", type=str, help="Teacher model name")
    parser.add_argument("--distillation_type", default="none", choices=["none", "soft", "hard"], help="Distillation type")
    parser.add_argument("--distillation_alpha", default=0.5, type=float, help="Distillation loss weight")
    parser.add_argument("--distillation_tau", default=1.0, type=float, help="Distillation temperature")

    # Distributed training
    parser.add_argument("--world_size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="Distributed training URL")

    parser.add_argument("--push_pull", action='store_true')
    parser.add_argument("--lateral_inhibition", action='store_true')

    # WandB 
    parser.add_argument('--wandb', action='store_false', help='USe wandb by default. Trigger to disable wandb')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)

    # Custom Transformers
    parser.add_argument("--stage1", type=str, default="lidiff", choices=transformer_dict.keys())
    parser.add_argument("--stage2", type=str, default="pp", choices=transformer_dict.keys())
    parser.add_argument("--stage3", type=str, default="spiking", choices=transformer_dict.keys())


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    wandb_log = misc.is_main_process()
    if global_rank == 0:
        wandb.login(key="27656ca0b0297d3f09e317d7a47bd97275cc33a1")
        wandb.init(
            project="LIT",
            name=args.name,
            id=wandb.util.generate_id(),
            tags=args.wandb_tags,
            resume='auto',
            config=vars(args),
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )
    
    # Step 3: Build list of transformer classes
    transformer_classes = [
        transformer_dict[args.stage1],
        transformer_dict[args.stage2],
        transformer_dict[args.stage3]
    ]


    if args.model_mode == "ms":
        # if args.dataset == "CIFAR10":
        #     model = models_new.__dict__[args.model](img_size=32, num_classes=10, push_pull=args.push_pull, lateral_inhibition=args.lateral_inhibition)
        #     print('Loaded CIFAR model')
        # else:
        model = QKFormer_10_768(T=args.time_steps, transformer_classes=transformer_classes)
    
    model.T = args.time_steps

    if  args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint["model"]
        # state_dict = model.state_dict()
        # for k in ["head.weight", "head.bias"]:
        #     if (
        #         k in checkpoint_model
        #         and checkpoint_model[k].shape != state_dict[k].shape
        #     ):
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]  # T=4注释

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)  # T=4注释

    model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    param_mem = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    print(f"Model parameters memory: {param_mem:.2f} MB")

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        # no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )

    if args.dataset == "CIFAR10":
        # AdamW optimizer with lr of 1e-3 and weight decay of 6e-2
        optimizer = AdamW(param_groups, lr=1e-3, weight_decay=6e-2)
    
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        lr_scheduler = None
    
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.kd:
        teacher_model = None
        if args.distillation_type == "none":
            args.distillation_type = "hard"
        print(f"Creating teacher model: {args.teacher_model}")
        # teacher_model_name = importlib.import_module("metaformer."+args.teacher_model)
        from metaformer import caformer_b36_in21ft1k

        teacher_model = caformer_b36_in21ft1k(pretrained=True)
        teacher_model.to(device)
        teacher_model.eval()
        # wrap the criterion in our custom DistillationLoss, which
        # just dispatches to the original criterion if args.distillation_type is 'none'
        criterion = DistillationLoss(
            criterion,
            teacher_model,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
        )

    print("criterion = %s" % str(criterion))

    # misc.load_model(
    #     args=args,
    #     model_without_ddp=model_without_ddp,
    #     optimizer=optimizer,
    #     loss_scaler=loss_scaler,
    # )

    if args.eval:

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

        clean_acc = test_stats['acc1']

        resultsTable = wandb.Table(columns=["corruption", "accuracy"])

        if args.c_eval:

            c_dataset_class = CIFAR10CDataset

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616]),
            ])

            severities = (1,2,3,4,5)

            corruption_to_accuracies = defaultdict(list)

            for corruption in c_dataset_class.corruptions:
                for severity in severities:
                    c_s_dst = c_dataset_class(
                        '/tmp/dataset/', transform=test_transform, severity=severity, corruption=corruption
                    )

                    corrupt_sampler = torch.utils.data.distributed.DistributedSampler(c_s_dst, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False)

                    c_s_loader = torch.utils.data.DataLoader(
                                    c_s_dst, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, sampler=corrupt_sampler)

                    test_stats = evaluate(c_s_loader, model, device)

                    acc = test_stats['acc1']  # Top-1 accuracy
        
                    corruption_to_accuracies[corruption].append(acc)

            # Step 3: after the loop, calculate averages
            corruption_to_avg_acc = {}

            for corruption, acc_list in corruption_to_accuracies.items():
                avg_acc = sum(acc_list) / len(acc_list)
                corruption_to_avg_acc[corruption] = avg_acc
            
            # Now add the overall average across corruptions
            all_corruption_avg = sum(corruption_to_avg_acc.values()) / len(corruption_to_avg_acc)
            
            resultsTable.add_data('Clean', clean_acc)
            resultsTable.add_data('Average', all_corruption_avg)        

            for corruption, avg_acc in corruption_to_avg_acc.items():
                resultsTable.add_data(corruption, avg_acc)

            if global_rank == 0:
                wandb.log({'corruption_table': resultsTable})

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
        try:
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                mixup_fn,
                args=args
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("CUDA OOM error detected:")
                print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
            else:
                raise e  # Re-raise if it's not a memory error
        if (epoch % 50 == 0 or epoch + 1 == args.epochs):
            print("Saving model at epoch:", epoch)
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                best_model=False
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        if test_stats["acc1"] > best_acc:
            print("Saving model at epoch:", epoch)
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                best_model=True
            )

            if wandb_log:
                wandb.log({
                    "test/acc1": test_stats["acc1"],
                    "test/acc5": test_stats["acc5"],
                    "test/loss": test_stats["loss"],
                    "epoch": epoch,
                })

                # if args.push_pull:
                #     for idx, block in enumerate(model.module.stage2_blocks):
                #         wandb.log(f'PushPull/Block{idx}_Push', block.attn.softplus(block.attn.push_w).item(), epoch)
                #         wandb.log(f'PushPull/Block{idx}_Pull', block.attn.softplus(block.attn.pull_w).item(), epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
