# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Modified for Fine-tuning ONLY the x2 block on a subset of classes.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from pathlib import Path

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from checkpoint_io import atomic_torch_save, get_resume_position, load_torch_checkpoint
from download import find_model


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        if args.resume is not None and Path(args.resume).parent.name == "checkpoints":
            checkpoint_dir = str(Path(args.resume).resolve().parent)
            experiment_dir = str(Path(checkpoint_dir).parent)
        else:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-x2-finetune"
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        if args.resume is not None:
            logger.info(f"Resuming experiment in {experiment_dir}")
        else:
            logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    
    # Load base pre-trained DiT model if provided
    if args.pretrained_ckpt is not None:
        logger.info(f"Loading base pre-trained model from {args.pretrained_ckpt}...")
        base_state_dict = find_model(args.pretrained_ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(base_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading base model (will use random init): {len(missing_keys)} keys")
            # Filter out x2-related keys as they might not exist in base model
            x2_missing = [k for k in missing_keys if 'x2' in k]
            other_missing = [k for k in missing_keys if 'x2' not in k]
            if other_missing:
                logger.warning(f"Non-x2 missing keys: {other_missing[:5]}...")  # Show first 5
            if x2_missing:
                logger.info(f"x2-related keys missing (expected): {len(x2_missing)} keys")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in base model: {len(unexpected_keys)} keys")
        logger.info("✓ Base pre-trained model loaded successfully")
    else:
        logger.warning("⚠️  No pre-trained checkpoint provided! Model will start with random weights.")
        logger.warning("⚠️  Fine-tuning without base weights will result in poor generation quality.")
    
    # --- FREEZING LOGIC START ---
    logger.info("Freezing main DiT model...")
    for p in model.parameters():
        p.requires_grad = False

    logger.info("Unfreezing x2 components...")
    # Unfreeze x2_embedder
    logger.info("  - Unfreezing x2_embedder...")
    for p in model.x2_embedder.parameters():
        p.requires_grad = True
    
    # Unfreeze learned aggregation from four x2 tokens to one main-stream token
    logger.info("  - Unfreezing x2_group_projection...")
    for p in model.x2_group_projection.parameters():
        p.requires_grad = True
    
    # Unfreeze x2_vit_block
    logger.info("  - Unfreezing x2_vit_block...")
    for p in model.x2_vit_block.parameters():
        p.requires_grad = True
        
    # Unfreeze projections if they exist
    if model.x2_vit_proj_in is not None:
        logger.info("  - Unfreezing x2_vit_proj_in...")
        for p in model.x2_vit_proj_in.parameters():
            p.requires_grad = True
            
    if model.x2_vit_proj_out is not None:
        logger.info("  - Unfreezing x2_vit_proj_out...")
        for p in model.x2_vit_proj_out.parameters():
            p.requires_grad = True
    
    # Unfreeze final_layer to allow output adaptation
    logger.info("  - Unfreezing final_layer...")
    for p in model.final_layer.parameters():
        p.requires_grad = True
    
    # Count trainable parameters by component
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Breakdown of trainable parameters
    x2_embedder_params = sum(p.numel() for p in model.x2_embedder.parameters() if p.requires_grad)
    x2_group_projection_params = sum(
        p.numel() for p in model.x2_group_projection.parameters() if p.requires_grad
    )
    x2_vit_block_params = sum(p.numel() for p in model.x2_vit_block.parameters() if p.requires_grad)
    x2_proj_in_params = sum(p.numel() for p in model.x2_vit_proj_in.parameters() if p.requires_grad) if model.x2_vit_proj_in is not None else 0
    x2_proj_out_params = sum(p.numel() for p in model.x2_vit_proj_out.parameters() if p.requires_grad) if model.x2_vit_proj_out is not None else 0
    final_layer_params = sum(p.numel() for p in model.final_layer.parameters() if p.requires_grad)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Trainable Parameters Breakdown:")
    logger.info(f"  x2_embedder:        {x2_embedder_params:>12,}")
    logger.info(f"  x2_group_projection:{x2_group_projection_params:>12,}")
    logger.info(f"  x2_vit_block:       {x2_vit_block_params:>12,}")
    logger.info(f"  x2_vit_proj_in:     {x2_proj_in_params:>12,}")
    logger.info(f"  x2_vit_proj_out:    {x2_proj_out_params:>12,}")
    logger.info(f"  final_layer:        {final_layer_params:>12,}")
    logger.info(f"  {'-'*60}")
    logger.info(f"  TOTAL TRAINABLE:    {trainable_params:>12,} / {total_params:,} ({trainable_params/total_params:.2%})")
    logger.info(f"{'='*60}\n")
    # --- FREEZING LOGIC END ---

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # Only pass trainable parameters to the optimizer
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # --- DATASET FILTERING START ---
    full_dataset = ImageFolder(args.data_path, transform=transform)
    
    # Filter dataset for selected classes
    logger.info(f"Filtering dataset for classes: {args.classes}")
    selected_indices = [i for i, label in enumerate(full_dataset.targets) if label in args.classes]
    
    if len(selected_indices) == 0:
        raise ValueError(f"No images found for classes {args.classes}. Check your dataset or class indices.")
        
    dataset = Subset(full_dataset, selected_indices)
    logger.info(f"Filtered Dataset contains {len(dataset):,} images (from {len(full_dataset)} total)")
    # --- DATASET FILTERING END ---

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Prepare models for training:
    train_steps = 0
    start_epoch = 0
    start_step_in_epoch = 0
    best_loss = float('inf')
    ema_restored = False

    if args.resume is not None:
        logger.info(f"Loading training state from {args.resume}...")
        resume_checkpoint = load_torch_checkpoint(args.resume, map_location="cpu")
        if "model" not in resume_checkpoint:
            raise ValueError(f"Resume checkpoint {args.resume} does not contain a 'model' state")

        missing_keys, unexpected_keys = model.module.load_state_dict(
            resume_checkpoint["model"],
            strict=False,
        )
        if unexpected_keys:
            raise ValueError(f"Unexpected model keys in resume checkpoint: {unexpected_keys[:5]}")
        logger.info(
            f"Loaded model state ({len(resume_checkpoint['model']):,} tensors; "
            f"{len(missing_keys):,} frozen/base tensors retained)"
        )

        if "ema" in resume_checkpoint:
            ema.load_state_dict(resume_checkpoint["ema"], strict=True)
            ema_restored = True
        else:
            logger.warning("Resume checkpoint has no EMA state; initializing EMA from the resumed model")

        if "opt" not in resume_checkpoint:
            raise ValueError(f"Resume checkpoint {args.resume} does not contain optimizer state")
        opt.load_state_dict(resume_checkpoint["opt"])

        train_steps, start_epoch, start_step_in_epoch = get_resume_position(
            resume_checkpoint,
            len(loader),
        )
        if resume_checkpoint.get("best", False):
            best_loss = float(resume_checkpoint.get("loss", best_loss))
        best_loss = float(resume_checkpoint.get("best_loss", best_loss))
        logger.info(
            f"Resume complete: global step {train_steps:,}, epoch {start_epoch}, "
            f"batch {start_step_in_epoch}/{len(loader)}"
        )

    if not ema_restored:
        update_ema(ema, model.module, decay=0)  # Initialize EMA with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training through epoch {args.epochs - 1} ({args.epochs} total epochs)...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        print(f"Beginning epoch {epoch}...")
        batches_to_skip = start_step_in_epoch if epoch == start_epoch else 0
        if batches_to_skip:
            logger.info(f"Skipping {batches_to_skip} already-completed batches in epoch {epoch}")

        for batch_index, (x, y) in enumerate(loader):
            if batch_index < batches_to_skip:
                continue
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
            print(f"Loss: {loss.item()}")
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Force print to console (bypasses logger buffering in Colab)
                print(f"✓ Step {train_steps:07d} | Loss: {avg_loss:.4f} | Steps/Sec: {steps_per_sec:.2f}", flush=True)
                
                # Save best checkpoint if loss improved
                if rank == 0 and avg_loss < best_loss:
                    best_loss = avg_loss
                    # Only save weights that were trainable to save space
                    trainable_keys = [k for k, p in model.module.named_parameters() if p.requires_grad]
                    model_state = {k: v for k, v in model.module.state_dict().items() if k in trainable_keys}

                    checkpoint = {
                        "model": model_state,
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "x2_finetune_only": True,
                        "step": train_steps,
                        "epoch": epoch,
                        "step_in_epoch": batch_index + 1,
                        "loss": avg_loss,
                        "best_loss": best_loss,
                        "best": True
                    }
                    best_checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}-loss-{best_loss:.4f}.pt"
                    atomic_torch_save(checkpoint, best_checkpoint_path)
                    logger.info(f"Saved BEST checkpoint to {best_checkpoint_path} (Loss: {avg_loss:.4f})")
                    print(f"🏆 New best checkpoint saved! Loss: {avg_loss:.4f}", flush=True)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint (periodic):
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # Only save weights that were trainable to save space
                    trainable_keys = [k for k, p in model.module.named_parameters() if p.requires_grad]
                    model_state = {k: v for k, v in model.module.state_dict().items() if k in trainable_keys}

                    checkpoint = {
                        "model": model_state,
                        # We save full EMA for now to be safe, or we could also just save partial EMA if needed
                        # But EMA usually keeps track of everything. To be safe for inference, let's keep full EMA or just trainable parts.
                        # For simplicity in this specialized script, let's stick to full EMA so inference scripts don't break,
                        # UNLESS the user explicitly wants lightweight.
                        # Given the user asked for LoRA-like "efficient" storage, best to save only what changed.
                        # But standard inference scripts expect full state dict.
                        # Compromise: Save full EMA (for immediate use) but partial model (for resume/space).
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "x2_finetune_only": True,
                        "step": train_steps,
                        "epoch": epoch,
                        "step_in_epoch": batch_index + 1,
                        "best_loss": best_loss
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    atomic_torch_save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path} (model contains only trainable parameters)")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    
    # New arguments
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="List of ImageNet class indices to train on (default: 0-9)")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, 
                        help="Path to base pre-trained DiT checkpoint (e.g., DiT-XL-2-256x256.pt). Required for proper fine-tuning.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Training checkpoint to resume, restoring model, EMA, optimizer, epoch, and global step.")

    args = parser.parse_args()
    main(args)
