# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}
X2_ADAPTER_CHECKPOINT_FORMAT = "x2_adapter_peft_v1"


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_format") == X2_ADAPTER_CHECKPOINT_FORMAT:
            return load_x2_adapter_checkpoint(checkpoint)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def load_x2_adapter_checkpoint(checkpoint):
    """
    Merge a lightweight x2 PEFT checkpoint into its recorded base DiT state dict.
    """
    base_ckpt = checkpoint.get("base_ckpt")
    base_ckpt_abs = checkpoint.get("base_ckpt_abs")
    if base_ckpt and base_ckpt not in pretrained_models and not os.path.isfile(base_ckpt) and base_ckpt_abs:
        base_ckpt = base_ckpt_abs
    if not base_ckpt:
        raise ValueError("x2 adapter checkpoints require a non-empty 'base_ckpt' field for inference.")

    base_state = find_model(base_ckpt).copy()
    adapter_state = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(adapter_state, dict):
        raise ValueError("x2 adapter checkpoint is missing a valid 'ema' or 'model' state dict.")

    base_state.update(adapter_state)
    return base_state


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage, weights_only=False)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
