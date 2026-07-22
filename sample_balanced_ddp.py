# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Generate an exactly class-balanced image batch with DiT/DuoDiT using DDP.

The generated PNGs live in one flat run directory. Their class labels are
preserved in filenames, a CSV manifest, metadata JSON, and ``arr_1`` of the
ADM-compatible NPZ. OpenAI's ADM evaluator continues to read images from
``arr_0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm


SAMPLE_NAME_PATTERN = re.compile(r"^(?P<index>\d{6,})-class(?P<class_id>\d{4,})\.png$")


@dataclass(frozen=True)
class SampleRecord:
    index: int
    filename: str
    class_id: int


def normalize_classes(classes: Iterable[int] | None, num_classes: int) -> list[int]:
    """Validate, deduplicate, and sort the requested class IDs."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    normalized = list(range(num_classes)) if classes is None else sorted(set(int(cls) for cls in classes))
    if not normalized:
        raise ValueError("classes must contain at least one class index")
    invalid = [cls for cls in normalized if cls < 0 or cls >= num_classes]
    if invalid:
        raise ValueError(f"Invalid class indices for num_classes={num_classes}: {invalid}")
    return normalized


def validate_balanced_request(num_samples: int, classes: Sequence[int]) -> int:
    """Return samples per class, rejecting requests that cannot be exactly balanced."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if not classes:
        raise ValueError("classes must contain at least one class index")
    if num_samples % len(classes) != 0:
        raise ValueError(
            f"num_samples={num_samples} is not divisible by the selected class count "
            f"({len(classes)}). Exact class balance is required."
        )
    return num_samples // len(classes)


def class_for_index(index: int, classes: Sequence[int]) -> int:
    """Map a global sample index to its deterministic round-robin class."""
    if index < 0:
        raise ValueError("index cannot be negative")
    if not classes:
        raise ValueError("classes must contain at least one class index")
    return int(classes[index % len(classes)])


def sample_filename(index: int, class_id: int) -> str:
    """Return the flat output filename for one generated sample."""
    if index < 0:
        raise ValueError("index cannot be negative")
    if class_id < 0:
        raise ValueError("class_id cannot be negative")
    return f"{index:06d}-class{class_id:04d}.png"


def parse_sample_filename(filename: str) -> SampleRecord:
    """Parse and validate a generated filename."""
    match = SAMPLE_NAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid balanced sample filename: {filename}")
    index = int(match.group("index"))
    class_id = int(match.group("class_id"))
    return SampleRecord(index=index, filename=filename, class_id=class_id)


def build_sample_records(num_samples: int, classes: Sequence[int]) -> list[SampleRecord]:
    """Build the canonical, index-ordered sample manifest."""
    validate_balanced_request(num_samples, classes)
    return [
        SampleRecord(
            index=index,
            filename=sample_filename(index, class_for_index(index, classes)),
            class_id=class_for_index(index, classes),
        )
        for index in range(num_samples)
    ]


def indices_for_rank(num_samples: int, rank: int, world_size: int) -> list[int]:
    """Partition exact global indices across ranks without padding."""
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative")
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, num_samples, world_size))


def batched(items: Sequence[int], batch_size: int) -> Iterator[Sequence[int]]:
    """Yield consecutive, possibly short final batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def format_class_label(classes: Sequence[int], num_classes: int) -> str:
    """Return a compact class-set label for the run directory."""
    if len(classes) == num_classes and list(classes) == list(range(num_classes)):
        return "all"
    if len(classes) <= 8:
        return "-".join(str(cls) for cls in classes)
    return f"{classes[0]}-{classes[-1]}-n{len(classes)}"


def checkpoint_label(checkpoint: str | os.PathLike[str]) -> str:
    """Return a filesystem-safe checkpoint label."""
    label = Path(checkpoint).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "checkpoint"


def build_run_name(args: argparse.Namespace, classes: Sequence[int]) -> str:
    """Build the single configuration-named output directory."""
    model_name = args.model.replace("/", "-")
    return (
        f"{model_name}-{checkpoint_label(args.ckpt)}-x2vit-{getattr(args, 'x2_vit_depth', 1)}b-"
        f"size-{args.image_size}-vae-{args.vae}-"
        f"steps-{args.num_sampling_steps}-cfg-{args.cfg_scale:g}-"
        f"classes-{format_class_label(classes, args.num_classes)}-"
        f"samples-{args.num_samples}-seed-{args.global_seed}"
    )


def assert_output_available(run_dir: Path, npz_path: Path) -> None:
    """Reject existing artifacts so separate runs cannot be mixed."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    if npz_path.exists():
        raise FileExistsError(f"Run NPZ already exists: {npz_path}")


def write_manifest(run_dir: Path, records: Sequence[SampleRecord]) -> Path:
    """Write the canonical sample-to-class mapping."""
    manifest_path = run_dir / "samples.csv"
    temporary_path = manifest_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["index", "filename", "class_id"])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    temporary_path.replace(manifest_path)
    return manifest_path


def validate_generated_samples(
    run_dir: Path,
    records: Sequence[SampleRecord],
) -> dict[int, int]:
    """Verify exact filenames and return the generated class histogram."""
    expected_names = [record.filename for record in records]
    actual_records = [parse_sample_filename(path.name) for path in run_dir.glob("*.png")]
    actual_names = [record.filename for record in sorted(actual_records, key=lambda record: record.index)]
    if actual_names != expected_names:
        expected_set = set(expected_names)
        actual_set = set(actual_names)
        missing = sorted(expected_set - actual_set)[:10]
        unexpected = sorted(actual_set - expected_set)[:10]
        raise ValueError(
            "Generated PNG set does not match the expected balanced manifest. "
            f"Missing examples: {missing}; unexpected examples: {unexpected}"
        )
    histogram = Counter(record.class_id for record in records)
    if len(set(histogram.values())) != 1:
        raise ValueError(f"Generated class counts are not equal: {dict(sorted(histogram.items()))}")
    return dict(sorted(histogram.items()))


def create_labeled_npz(
    run_dir: Path,
    records: Sequence[SampleRecord],
    npz_path: Path | None = None,
) -> Path:
    """
    Build a memory-bounded ADM-compatible NPZ.

    ``arr_0`` contains index-ordered uint8 RGB images and ``arr_1`` contains the
    matching int64 class IDs. Temporary image storage is a NumPy memory map.
    """
    if not records:
        raise ValueError("records must contain at least one sample")
    ordered_records = sorted(records, key=lambda record: record.index)
    expected_indices = list(range(len(ordered_records)))
    actual_indices = [record.index for record in ordered_records]
    if actual_indices != expected_indices:
        raise ValueError("records must have contiguous indices starting at zero")

    npz_path = npz_path or Path(f"{run_dir}.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    published = False

    try:
        first_path = run_dir / ordered_records[0].filename
        with Image.open(first_path) as first_image:
            first_array = np.asarray(first_image.convert("RGB"), dtype=np.uint8)
        if first_array.ndim != 3 or first_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image at {first_path}, found shape {first_array.shape}")

        with tempfile.TemporaryDirectory(prefix=".balanced-npz-", dir=npz_path.parent) as temporary_dir:
            images_path = Path(temporary_dir) / "images.npy"
            images = np.lib.format.open_memmap(
                images_path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(ordered_records), *first_array.shape),
            )
            labels = np.empty(len(ordered_records), dtype=np.int64)

            for position, record in enumerate(tqdm(ordered_records, desc="Building labeled NPZ")):
                image_path = run_dir / record.filename
                with Image.open(image_path) as image:
                    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if image_array.shape != first_array.shape:
                    raise ValueError(
                        f"Image shape mismatch at {image_path}: "
                        f"expected {first_array.shape}, found {image_array.shape}"
                    )
                images[position] = image_array
                labels[position] = record.class_id

            images.flush()
            with temporary_npz.open("wb") as output:
                np.savez(output, arr_0=images, arr_1=labels)
            del images

        temporary_npz.replace(npz_path)
        published = True
    finally:
        if not published:
            temporary_npz.unlink(missing_ok=True)
    return npz_path


def write_metadata(
    run_dir: Path,
    args: argparse.Namespace,
    classes: Sequence[int],
    class_counts: dict[int, int],
    world_size: int,
    npz_path: Path,
) -> Path:
    """Write reproducibility metadata after all artifacts have been validated."""
    metadata = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "image_size": args.image_size,
        "vae": args.vae,
        "num_classes": args.num_classes,
        "classes": list(classes),
        "num_samples": args.num_samples,
        "samples_per_class": args.num_samples // len(classes),
        "class_counts": {str(class_id): count for class_id, count in class_counts.items()},
        "num_sampling_steps": args.num_sampling_steps,
        "cfg_scale": args.cfg_scale,
        "global_seed": args.global_seed,
        "tf32": args.tf32,
        "world_size": world_size,
        "per_proc_batch_size": args.per_proc_batch_size,
        "sample_directory": str(run_dir.resolve()),
        "manifest": str((run_dir / "samples.csv").resolve()),
        "npz": str(npz_path.resolve()),
        "npz_arrays": {
            "arr_0": "uint8 RGB images in global-index order",
            "arr_1": "int64 class IDs aligned with arr_0",
        },
    }
    metadata_path = run_dir / "metadata.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)
    return metadata_path


def validate_cli_args(args: argparse.Namespace) -> list[int]:
    """Validate arguments before initializing distributed CUDA."""
    classes = normalize_classes(args.classes, args.num_classes)
    validate_balanced_request(args.num_samples, classes)
    if args.per_proc_batch_size < 1:
        raise ValueError("per_proc_batch_size must be at least 1")
    if args.cfg_scale < 1.0:
        raise ValueError("cfg_scale must be at least 1.0")
    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8")
    checkpoint_path = Path(args.ckpt).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return classes


def run_sampling(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run exact balanced sampling and return the PNG directory and NPZ path."""
    classes = validate_cli_args(args)
    run_name = build_run_name(args, classes)
    run_dir = Path(args.sample_dir).expanduser() / run_name
    npz_path = Path(f"{run_dir}.npz")
    assert_output_available(run_dir, npz_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Balanced DDP sampling requires at least one CUDA GPU")

    # Heavy model dependencies stay out of helper-only imports and unit tests.
    from diffusers.models import AutoencoderKL

    from diffusion import create_diffusion
    from download import find_model
    from models import DiT_models

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    rank_seed = args.global_seed * world_size + rank
    torch.manual_seed(rank_seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={rank_seed}, world_size={world_size}.")

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving balanced PNG samples at {run_dir}")
    dist.barrier()

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        x2_vit_depth=args.x2_vit_depth,
    ).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    using_cfg = args.cfg_scale > 1.0

    rank_indices = indices_for_rank(args.num_samples, rank, world_size)
    rank_batches = list(batched(rank_indices, args.per_proc_batch_size))
    progress: Iterable[Sequence[int]] = rank_batches
    if rank == 0:
        progress = tqdm(rank_batches, desc="Sampling rank 0")

    for batch_indices in progress:
        batch_size = len(batch_indices)
        class_ids = [class_for_index(index, classes) for index in batch_indices]
        z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
        y = torch.tensor(class_ids, device=device, dtype=torch.long)

        if using_cfg:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((batch_size,), args.num_classes, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = {"y": y}
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        decoded = vae.decode(samples / 0.18215).sample
        decoded = (
            torch.clamp(127.5 * decoded + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        for index, class_id, image_array in zip(batch_indices, class_ids, decoded):
            Image.fromarray(image_array).save(run_dir / sample_filename(index, class_id))

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        records = build_sample_records(args.num_samples, classes)
        class_counts = validate_generated_samples(run_dir, records)
        manifest_path = write_manifest(run_dir, records)
        generated_npz = create_labeled_npz(run_dir, records, npz_path)
        metadata_path = write_metadata(
            run_dir,
            args,
            classes,
            class_counts,
            world_size,
            generated_npz,
        )
        print(f"Manifest: {manifest_path}")
        print(f"Metadata: {metadata_path}")
        print(f"NPZ: {generated_npz}")
        print("Done.")

    return run_dir, npz_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, required=True, help="DuoDiT checkpoint containing EMA weights.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--sample-dir", type=str, default="balanced_samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Class IDs to balance exactly. Defaults to all classes.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--x2-vit-depth", type=int, choices=[1, 2, 4], default=1,
                        help="Must match the final ViT block count used during training.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmuls on supported GPUs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_sampling(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
