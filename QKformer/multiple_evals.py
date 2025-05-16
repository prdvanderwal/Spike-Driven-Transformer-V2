#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional
from collections import defaultdict
import sys
import math

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torchvision import datasets, transforms
import torch.distributed as dist


from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset, create_loader
# from loader import create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from saver import *
from snn_v2 import *
from datasets import CIFAR10CDataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
        print('Native amp')
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


# Step 1: Define string-to-class mapping
transformer_dict = {
    "lidiff": LiDiffTransformer,
    "pp": PPTransformer,
    "spiking": SpikingTransformer,
    "token": TokenSpikingTransformer,
    "pptoken": PushPullTransformer
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='cifar10.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments') # imagenet.yml  cifar10.yml

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Model detail
parser.add_argument('--model', default='vitsnn', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')


# Dataset / Model parameters
parser.add_argument('-data-dir', metavar='DIR',default="/media/data/spike-transformer-network/torch/cifar10/",
                    help='path to dataset') #./torch/imagenet/
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')

parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 32)')
# parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
#                     help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 1.0], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

# WandB 
parser.add_argument('--wandb', action='store_false', help='USe wandb by default. Trigger to disable wandb')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)

# Custom Transformers
parser.add_argument("--stage1", type=str, nargs='+', default="lidiff", choices=transformer_dict.keys())
parser.add_argument("--stage2", type=str, nargs='+', default="pp", choices=transformer_dict.keys())
parser.add_argument("--stage3", type=str, nargs='+', default="spiking", choices=transformer_dict.keys())

parser.add_argument('--current_epoch', type=int, default=0)
parser.add_argument('--experiment_name', type=str, nargs='+', default=None)
parser.add_argument('--resume_paths', type=str, nargs='+', default=None)
parser.add_argument('--spike_rate', action='store_true', help='Calculate the spike rate of the model on eval set')

args = parser.parse_args()


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:1'
    args.world_size = 1
    args.rank = 0  # global rank


    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(f"Training in distributed mode. Rank {args.rank} of {args.world_size} total.")
    else:
        args.local_rank = 0
        args.device = "cuda:0"
        args.world_size = 1
        args.rank = 0
        _logger.info("Training in single-GPU mode.")

    assert args.rank >= 0
       
    if args.log_wandb and args.rank ==0:
        if has_wandb:
            wandb.login(key="27656ca0b0297d3f09e317d7a47bd97275cc33a1")
            wandb.init(
                project="LIT",
                name=args.name,
                id=wandb.util.generate_id(),
                tags=args.wandb_tags,
                resume='auto',
                config=vars(args),
            )
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")


    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.spike_rate:
        final_table = wandb.Table(columns=["Model", "Acc1", "Acc5", "Loss"])

    for path_idx in range(len(args.resume_paths)):

        args.path_idx = path_idx

        # Step 3: Build list of transformer classes
        transformer_classes = [
            transformer_dict[args.stage1[path_idx]],
            transformer_dict[args.stage2[path_idx]],
            transformer_dict[args.stage3[path_idx]]
        ]


        model = create_model(
            "QKFormer",
            pretrained=False,
            drop_rate=0.,
            drop_path_rate=0.2,
            drop_block_rate=None,
            img_size_h=args.img_size, img_size_w=args.img_size,
            patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
            in_channels=3, num_classes=args.num_classes, qkv_bias=False,
            depths=args.layer, sr_ratios=1,
            T=args.time_step, transformer_classes=transformer_classes, spike_rate=args.spike_rate
        )


        print("Creating model")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"number of params: {n_parameters}")

        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

        if args.local_rank == 0:
            _logger.info(
                f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    
        model.cuda()

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if use_amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            loss_scaler = ApexScaler()
            if args.local_rank == 0:
                _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()
            if args.local_rank == 0:
                _logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if args.local_rank == 0:
                _logger.info('AMP not enabled. Training in float32.')

        resume_epoch = resume_checkpoint(
            model, args.resume_paths[path_idx],
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
        args.current_epoch = resume_epoch

        print("Model = %s" % str(model))

        # setup distributed training
        if args.distributed:
            if has_apex and use_amp != 'native':
                # Apex DDP preferred unless native amp is activated
                if args.local_rank == 0:
                    _logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(model, delay_allreduce=True)
            else:
                if args.local_rank == 0:
                    _logger.info("Using native Torch DistributedDataParallel.")
                model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
            # NOTE: EMA model does not need to be wrapped by DDP

        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size, download=False)


        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )

        validate_loss_fn = nn.CrossEntropyLoss().cuda()

        if args.spike_rate:
            
            test_stats = spike_rate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            final_table.add_data(args.experiment_name[args.path_idx],test_stats['top1'], test_stats['top5'], test_stats['loss'])

            if path_idx == len(args.resume_paths) - 1 and args.local_rank == 0:
                wandb.log({'Results table': final_table})
                wandb.finish()
                sys.exit(0)
            
            continue

        # Start evaluation
        test_stats = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
        print(
            f"Accuracy of the network on the {len(dataset_eval)} test images: {test_stats['top1']:.1f}%"
        )

        clean_acc = test_stats['top1']

        resultsTable = wandb.Table(columns=["corruption", "accuracy"])

        c_dataset_class = CIFAR10CDataset

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616]),
        ])

        severities = (1,2,3,4,5)

        corruption_to_accuracies = defaultdict(list)

        old_prefetcher = args.prefetcher
        args.prefetcher = False

        for corruption in c_dataset_class.corruptions:
            for severity in severities:
                c_s_dst = c_dataset_class(
                    '/tmp/dataset/', transform=test_transform, severity=severity, corruption=corruption
                )

                corrupt_sampler = torch.utils.data.distributed.DistributedSampler(c_s_dst, num_replicas=args.world_size, rank=args.local_rank, shuffle=False, drop_last=False)

                old_prefetcher = args.prefetcher
                args.prefetcher = False
                c_s_loader = torch.utils.data.DataLoader(
                                c_s_dst, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, sampler=corrupt_sampler)

                test_stats = validate(model, c_s_loader, validate_loss_fn, args, amp_autocast=amp_autocast, corrupt=True)
                
                acc = test_stats['top1']  # Top-1 accuracy

                corruption_to_accuracies[corruption].append(acc)
        
        args.prefetcher = old_prefetcher

        # Step 3: after the loop, calculate averages
        corruption_to_avg_acc = {}

        for corruption, acc_list in corruption_to_accuracies.items():
            avg_acc = sum(acc_list) / len(acc_list)
            corruption_to_avg_acc[corruption] = avg_acc
        
        # Now add the overall average across corruptions
        all_corruption_avg = sum(corruption_to_avg_acc.values()) / len(corruption_to_avg_acc)
        
        

        if args.local_rank == 0:
            resultsTable.add_data('Clean', clean_acc)
            resultsTable.add_data('Average', all_corruption_avg)        

            for corruption, avg_acc in corruption_to_avg_acc.items():
                resultsTable.add_data(corruption, avg_acc)
            wandb.log({f'Corruption_table {args.experiment_name[path_idx]}': resultsTable})



def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', corrupt=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
    
    if args.local_rank == 0:
        if not corrupt:
            log_data = {
                            f'test/acc1_{args.experiment_name[args.path_idx]}': top1_m.avg,
                            f'test/acc5_{args.experiment_name[args.path_idx]}': top5_m.avg,
                            f'test/loss_{args.experiment_name[args.path_idx]}': losses_m.avg
                        }
            wandb.log(log_data, step=args.current_epoch)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

def spike_rate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', corrupt=False):

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    total_spikes = defaultdict(float)
    total_elements = defaultdict(int)

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with amp_autocast(enabled=False):
                output, (attn1, mlp1, attn2, mlp2)  = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            spike_groups = {
            'stage1-attn': attn1,  # dict: qe, qi, k, attn, x
            'stage1-mlp': mlp1,  # dict: mlp1, mlp2
            'stage2-attn': attn2,
            'stage2-mlp': mlp2,  # dict: mlp1, mlp2
            }

            if math.isinf(attn1["k"].sum().item()):
                print('k is still infinite')
                sys.exit(0)

            for group_name, subdict in spike_groups.items():
                for layer_name, tensor in subdict.items():
                    key = f"{group_name}/{layer_name}"
                    total_spikes[key] += tensor.sum().item()
                    total_elements[key] += tensor.numel()


            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
    
    if args.distributed:
        for key in total_spikes:
            spike_tensor = torch.tensor(total_spikes[key], device=args.device, dtype=torch.float32)
            element_tensor = torch.tensor(total_elements[key], device=args.device, dtype=torch.float32)

            dist.all_reduce(spike_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(element_tensor, op=dist.ReduceOp.SUM)

            total_spikes[key] = spike_tensor.item()
            total_elements[key] = element_tensor.item()

    # Then compute the average
    avg_firing_rate = {
        key: total_spikes[key] / total_elements[key] for key in total_spikes
    }

    if args.local_rank == 0:
        columns = ["Layer", "Avg Firing Rate"]
        data = [[key, value] for key, value in avg_firing_rate.items()]
        table = wandb.Table(columns=columns, data=data)

        # Step 2: Log it with a dynamic key
        table_name = args.experiment_name[args.path_idx]
        wandb.log({table_name: table})

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics



if __name__ == '__main__':
    main()