"""
Reliable helpers for saving and loading PyTorch checkpoints.
"""

import argparse
import errno
import os
import pickle
import uuid
from pathlib import Path

import torch


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint file exists but cannot be deserialized."""


def _format_size(size_bytes):
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def load_torch_checkpoint(path, map_location="cpu"):
    """
    Load a checkpoint and report storage/corruption failures with useful context.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as exc:
        try:
            size = _format_size(checkpoint_path.stat().st_size)
        except OSError:
            size = "unknown size"

        if isinstance(exc, OSError) and exc.errno == errno.EIO:
            cause = (
                "the filesystem returned an input/output error. The checkpoint may "
                "be incomplete, or the storage device/network mount may be unhealthy"
            )
        else:
            cause = "the file is incomplete, corrupt, or not a supported PyTorch checkpoint"

        raise CheckpointLoadError(
            f"Could not load checkpoint '{checkpoint_path}' ({size}): {cause}. "
            f"Original error: {exc}"
        ) from exc


def get_resume_position(checkpoint, steps_per_epoch):
    """
    Return ``(train_steps, epoch, step_in_epoch)`` for old and new checkpoints.

    Older checkpoints only stored the global step, so derive their position from
    the current loader length. New checkpoints store the explicit position.
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")

    train_steps = int(checkpoint.get("step", 0))
    if train_steps < 0:
        raise ValueError("checkpoint step cannot be negative")

    if "epoch" in checkpoint and "step_in_epoch" in checkpoint:
        epoch = int(checkpoint["epoch"])
        step_in_epoch = int(checkpoint["step_in_epoch"])
        if epoch < 0 or step_in_epoch < 0:
            raise ValueError("checkpoint epoch position cannot be negative")
        completed_epochs, step_in_epoch = divmod(step_in_epoch, steps_per_epoch)
        epoch += completed_epochs
    else:
        epoch, step_in_epoch = divmod(train_steps, steps_per_epoch)

    return train_steps, epoch, step_in_epoch


def _fsync_directory(directory):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def atomic_torch_save(obj, path):
    """
    Save to a temporary file and atomically publish the completed checkpoint.

    If saving is interrupted or fails, an existing checkpoint at ``path`` remains
    untouched and the incomplete temporary file is removed.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    published = False

    try:
        torch.save(obj, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, checkpoint_path)
        published = True
        _fsync_directory(checkpoint_path.parent)
    finally:
        if not published:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _describe_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return type(checkpoint).__name__
    keys = ", ".join(sorted(str(key) for key in checkpoint))
    return f"dict keys: {keys}"


def main():
    parser = argparse.ArgumentParser(description="Verify that PyTorch checkpoints are readable.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to verify.")
    args = parser.parse_args()

    failed = False
    for checkpoint_path in args.checkpoints:
        try:
            checkpoint = load_torch_checkpoint(checkpoint_path)
        except (CheckpointLoadError, FileNotFoundError) as exc:
            failed = True
            print(f"FAILED: {exc}")
        else:
            print(f"OK: {checkpoint_path} ({_describe_checkpoint(checkpoint)})")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
