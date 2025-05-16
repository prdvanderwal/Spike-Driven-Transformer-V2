import os
import operator
import logging
import torch
from timm.utils import CheckpointSaver as TimmCheckpointSaver
from timm.utils.model import unwrap_model, get_state_dict
import wandb

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)

    wandb.save(filename)

    if is_best:
        best_filepath = os.path.join(path, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        wandb.save(best_filepath)

def build_checkpoint_state(model, optimizer, epoch, amp_scaler=None, model_ema=None, args=None):
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp_scaler': amp_scaler.state_dict() if amp_scaler else None,
        'model_ema': model_ema.state_dict() if model_ema else None,
        'args': args,
    }

def load_model(model, amp_scaler, model_ema, optimizer, path, args):
    # optionally resume from a checkpoint
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        if torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(path, map_location=loc)

            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            epoch = checkpoint['epoch']
            amp_scaler.load_state_dict(checkpoint['amp_scaler']) if amp_scaler else None
            optimizer.load_state_dict(checkpoint['optimizer']) if optimizer else None
            model_ema.load_state_dict(checkpoint['model_ema']) if model_ema else None
            print("=> loaded checkpoint '{}' (step {})"
                .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume_paths[path_idx]))