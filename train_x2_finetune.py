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
from torch.utils.data import DataLoader, Dataset, Subset
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
import hashlib
import json
import logging
import os

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

CHECKPOINT_FORMAT = "x2_adapter_peft_v1"


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def unwrap_compiled_model(model):
    """
    Return the original nn.Module when torch.compile wraps it in an OptimizedModule.
    """
    return getattr(model, "_orig_mod", model)


def get_trainable_keys(model):
    """
    Return parameter names that participate in PEFT updates.
    """
    model = unwrap_compiled_model(model)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def filtered_state_dict(model, keys):
    """
    Return a CPU state dict containing only selected parameter keys.
    """
    model = unwrap_compiled_model(model)
    state = model.state_dict()
    return {key: state[key].detach().cpu() for key in keys if key in state}


@torch.no_grad()
def update_ema(ema_model, model, trainable_keys=None, decay=0.9999):
    """
    Step the EMA model towards the current model for trainable PEFT params only.
    """
    model = unwrap_compiled_model(model)
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    keys = trainable_keys if trainable_keys is not None else model_params.keys()

    for name in keys:
        param = model_params[name]
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def build_peft_checkpoint(model, ema_model, opt, args, trainable_keys, step, loss=None, best=False):
    """
    Build a lightweight adapter-only checkpoint.
    """
    base_ckpt_abs = None
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        base_ckpt_abs = os.path.abspath(args.pretrained_ckpt)
    checkpoint = {
        "model": filtered_state_dict(model, trainable_keys),
        "ema": filtered_state_dict(ema_model, trainable_keys),
        "opt": opt.state_dict(),
        "args": args,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "base_ckpt": args.pretrained_ckpt,
        "base_ckpt_abs": base_ckpt_abs,
        "trainable_keys": list(trainable_keys),
        "x2_finetune_only": True,
        "step": step,
    }
    if loss is not None:
        checkpoint["loss"] = loss
    if best:
        checkpoint["best"] = True
    return checkpoint


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


class LatentCacheDataset(Dataset):
    """
    In-memory dataset backed by cached VAE latents and original ImageNet labels.
    """
    def __init__(self, cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.latents = payload["latents"]
        self.labels = payload["labels"].long()
        self.manifest = payload.get("manifest", {})

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.latents[index], self.labels[index]


def create_image_transform(image_size, random_flip):
    ops = [transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size))]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transforms.Compose(ops)


def dataloader_kwargs(args):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def latent_cache_path(args, full_dataset, selected_indices):
    data_path = os.path.abspath(args.data_path)
    sample_entries = []
    for index in selected_indices:
        path, label = full_dataset.samples[index]
        stat = os.stat(path)
        sample_entries.append({
            "path": os.path.relpath(path, data_path),
            "label": int(label),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })

    manifest = {
        "data_path": data_path,
        "image_size": args.image_size,
        "vae": args.vae,
        "classes": sorted(int(cls) for cls in args.classes),
        "samples": sample_entries,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(args.latent_cache_dir, f"latents-{digest}.pt"), manifest


@torch.no_grad()
def build_latent_cache(dataset, cache_path, manifest, vae, device, args, logger):
    os.makedirs(args.latent_cache_dir, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=args.latent_cache_batch_size,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs(args)
    )
    vae.eval()
    latents = []
    labels = []
    logger.info(f"Building latent cache at {cache_path}...")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
        latents.append(latent.cpu())
        labels.append(y.cpu())

    payload = {
        "latents": torch.cat(latents, dim=0),
        "labels": torch.cat(labels, dim=0).long(),
        "manifest": manifest,
    }
    tmp_path = f"{cache_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    logger.info(f"Saved latent cache with {payload['latents'].shape[0]:,} samples.")


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
    device_id = rank % torch.cuda.device_count()
    device = torch.device("cuda", device_id)
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-x2-finetune"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
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
    
    # Unfreeze x2_cls_tokens (LEARNABLE CLS TOKENS)
    if hasattr(model, 'x2_cls_tokens'):
        logger.info("  - Unfreezing x2_cls_tokens (learnable CLS tokens)...")
        model.x2_cls_tokens.requires_grad = True
        cls_token_params = model.x2_cls_tokens.numel()
        logger.info(f"    ✓ x2_cls_tokens shape: {model.x2_cls_tokens.shape}, parameters: {cls_token_params:,}")
        assert model.x2_cls_tokens.requires_grad, "x2_cls_tokens should be trainable!"
    else:
        logger.warning("  ⚠️  model.x2_cls_tokens not found! This should not happen.")
    
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
    x2_cls_token_params = model.x2_cls_tokens.numel() if hasattr(model, 'x2_cls_tokens') and model.x2_cls_tokens.requires_grad else 0
    x2_vit_block_params = sum(p.numel() for p in model.x2_vit_block.parameters() if p.requires_grad)
    x2_proj_in_params = sum(p.numel() for p in model.x2_vit_proj_in.parameters() if p.requires_grad) if model.x2_vit_proj_in is not None else 0
    x2_proj_out_params = sum(p.numel() for p in model.x2_vit_proj_out.parameters() if p.requires_grad) if model.x2_vit_proj_out is not None else 0
    final_layer_params = sum(p.numel() for p in model.final_layer.parameters() if p.requires_grad)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Trainable Parameters Breakdown:")
    logger.info(f"  x2_embedder:        {x2_embedder_params:>12,}")
    logger.info(f"  x2_cls_tokens:      {x2_cls_token_params:>12,}  ⭐ (LEARNABLE CLS TOKENS)")
    logger.info(f"  x2_vit_block:       {x2_vit_block_params:>12,}")
    logger.info(f"  x2_vit_proj_in:     {x2_proj_in_params:>12,}")
    logger.info(f"  x2_vit_proj_out:    {x2_proj_out_params:>12,}")
    logger.info(f"  final_layer:        {final_layer_params:>12,}")
    logger.info(f"  {'-'*60}")
    logger.info(f"  TOTAL TRAINABLE:    {trainable_params:>12,} / {total_params:,} ({trainable_params/total_params:.2%})")
    logger.info(f"{'='*60}\n")
    trainable_keys = get_trainable_keys(model)
    # --- FREEZING LOGIC END ---

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0 or newer.")
        logger.info(f"Compiling model with torch.compile(mode={args.compile_mode!r})...")
        model = torch.compile(model, mode=args.compile_mode)
    model = DDP(model, device_ids=[device_id])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # Only pass trainable parameters to the optimizer
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = create_image_transform(args.image_size, random_flip=not args.cache_latents)
    
    # --- DATASET FILTERING START ---
    full_dataset = ImageFolder(args.data_path, transform=transform)
    
    # Filter dataset for selected classes
    logger.info(f"Filtering dataset for classes: {args.classes}")
    selected_indices = [i for i, label in enumerate(full_dataset.targets) if label in args.classes]
    
    if len(selected_indices) == 0:
        raise ValueError(f"No images found for classes {args.classes}. Check your dataset or class indices.")
        
    image_dataset = Subset(full_dataset, selected_indices)
    logger.info(f"Filtered Dataset contains {len(image_dataset):,} images (from {len(full_dataset)} total)")
    # --- DATASET FILTERING END ---

    using_latent_cache = args.cache_latents
    vae = None
    if using_latent_cache:
        cache_path, cache_manifest = latent_cache_path(args, full_dataset, selected_indices)
        needs_cache_build = args.rebuild_latent_cache or not os.path.isfile(cache_path)
        if rank == 0:
            if needs_cache_build:
                vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
                build_latent_cache(image_dataset, cache_path, cache_manifest, vae, device, args, logger)
                del vae
                vae = None
            else:
                logger.info(f"Reusing latent cache at {cache_path}")
        dist.barrier()
        dataset = LatentCacheDataset(cache_path)
        logger.info(f"Latent cache contains {len(dataset):,} samples")
    else:
        dataset = image_dataset
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

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
        drop_last=True,
        **dataloader_kwargs(args)
    )

    # Prepare models for training:
    update_ema(ema, model.module, trainable_keys=trainable_keys, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_loss = float('inf')  # Track best loss for saving best checkpoint

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if using_latent_cache:
                flip_ids = torch.rand(x.shape[0], device=device) < 0.5
                if flip_ids.any():
                    x[flip_ids] = torch.flip(x[flip_ids], dims=[3])
            else:
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module, trainable_keys=trainable_keys)
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
                    checkpoint = build_peft_checkpoint(
                        model.module,
                        ema,
                        opt,
                        args,
                        trainable_keys,
                        train_steps,
                        loss=avg_loss,
                        best=True
                    )
                    
                    # Verify x2_cls_tokens is included in checkpoint
                    cls_token_in_checkpoint = 'x2_cls_tokens' in trainable_keys
                    if cls_token_in_checkpoint:
                        logger.info(f"✓ x2_cls_tokens included in checkpoint (shape: {checkpoint['model']['x2_cls_tokens'].shape})")
                    else:
                        logger.warning("⚠️  x2_cls_tokens NOT found in trainable keys - this should not happen!")

                    best_checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}-loss-{best_loss:.4f}.pt"
                    torch.save(checkpoint, best_checkpoint_path)
                    logger.info(f"Saved BEST checkpoint to {best_checkpoint_path} (Loss: {avg_loss:.4f})")
                    print(f"🏆 New best checkpoint saved! Loss: {avg_loss:.4f}", flush=True)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint (periodic):
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # Verify x2_cls_tokens is included in checkpoint
                    if 'x2_cls_tokens' in trainable_keys:
                        logger.info(f"✓ x2_cls_tokens included in periodic checkpoint")
                    checkpoint = build_peft_checkpoint(
                        model.module,
                        ema,
                        opt,
                        args,
                        trainable_keys,
                        train_steps
                    )
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path} (Model contains only trainable params, including x2_cls_tokens)")
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
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-latents", dest="cache_latents", action="store_true",
                             help="Precompute and reuse VAE latents for this dataset/class subset.")
    cache_group.add_argument("--no-cache-latents", dest="cache_latents", action="store_false",
                             help="Encode images through the VAE on every training step.")
    parser.set_defaults(cache_latents=True)
    parser.add_argument("--latent-cache-dir", type=str, default=".cache/duodit_latents")
    parser.add_argument("--rebuild-latent-cache", action="store_true")
    parser.add_argument("--latent-cache-batch-size", type=int, default=64)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the DiT model before DDP wrapping.")
    parser.add_argument("--compile-mode", type=str, default="default")
    
    # New arguments
    parser.add_argument("--classes", type=int, nargs="+", required=True, help="List of ImageNet class indices to train on (default: 0-9)")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, 
                        help="Path to base pre-trained DiT checkpoint (e.g., DiT-XL-2-256x256.pt). Required for proper fine-tuning.")

    args = parser.parse_args()
    main(args)
