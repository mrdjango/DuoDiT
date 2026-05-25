# DiT x2-Block Fine-Tuning Guide

This repository contains a specialized script `train_x2_finetune.py` for fine-tuning **only** the secondary `x2` input stream (embedding + ViT block + projections) of a DiT model while keeping the main backbone frozen.

## Overview

This approach is useful for adapting the DiT model to new conditioning signals (via the `x2` pathway) without destroying the high-quality image generation capabilities of the pre-trained DiT backbone.

**Key Features:**
*   **Frozen Backbone**: The core DiT transformer is frozen.
*   **Trainable x2 Branch**: Only `x2_embedder`, `x2_vit_block`, and associated projection layers are trained.
*   **Filtered Training**: Supports training on a specific subset of ImageNet classes (e.g., just 10 classes) while maintaining correct class embedding indices.
*   **Efficient Checkpoints**: Saves only the trainable parameters, significantly reducing checkpoint size.

## Usage

### 1. Training

Run the training script using `torchrun`. You can specify the ImageNet class indices you want to train on using the `--classes` argument.

**Example 1: Train on the first 10 classes**
```bash
torchrun --nnodes=1 --nproc_per_node=8 train_x2_finetune.py \
    --model DiT-XL/2 \
    --data-path /path/to/imagenet/train \
    --classes 0 1 2 3 4 5 6 7 8 9 \
    --batch-size 32 \
    --epochs 100
```

**Example 2: Train on a single specific class (e.g., index 985 "Daisy")**
```bash
torchrun --nnodes=1 --nproc_per_node=8 train_x2_finetune.py \
    --model DiT-XL/2 \
    --data-path /path/to/imagenet/train \
    --classes 985
```

### 2. Monitoring
Logs are saved to `results/xxx-DiT-XL-2-x2-finetune/log.txt`.
You can monitor training progress (Loss, Steps/Sec) in the console or log file.

### 3. Checkpoints
To save storage space, the checkpoints saved in `results/.../checkpoints/*.pt` contain **only the trainable parameters** (the `x2` branch logic).

To load these checkpoints for inference or resuming, you will need to:
1.  Initialize a standard DiT model.
2.  Load the checkpoint with `strict=False` (since it's missing the frozen backbone weights).
    *   *Note: The script currently saves the full EMA state dict for valid inference out-of-the-box, but the 'model' key inside the checkpoint only has trainable weights.*

## Verification
A test script `test_freezing.py` is included to verify that the freezing logic is working correctly (backbone gradients disabled, x2 gradients enabled).

```bash
python test_freezing.py
```
