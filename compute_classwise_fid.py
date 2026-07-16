#!/usr/bin/env python3
"""Compute clean-fid separately for classes encoded in sample filenames."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_CLASS_PATTERN = re.compile(
    r"(?:^|[-_.])class[-_]?(?P<class_name>[A-Za-z0-9]+)(?=$|[-_.])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RealClass:
    key: str
    name: str
    directory: Path


def canonical_class_key(value: str) -> str:
    """Normalize numeric labels while keeping named labels case-insensitive."""
    value = value.strip()
    if not value:
        raise ValueError("Class names cannot be empty")
    if value.isdecimal():
        return str(int(value))
    return value.casefold()


def iter_image_files(directory: Path) -> list[Path]:
    """Return all supported image files below a directory."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def discover_real_classes(real_root: Path) -> dict[str, RealClass]:
    """Discover one class per direct child directory."""
    if not real_root.is_dir():
        raise ValueError(f"Real dataset directory does not exist: {real_root}")

    classes: dict[str, RealClass] = {}
    for directory in sorted(path for path in real_root.iterdir() if path.is_dir()):
        key = canonical_class_key(directory.name)
        if key in classes:
            other = classes[key].directory
            raise ValueError(
                f"Real class folders {other} and {directory} normalize to the same class key"
            )
        classes[key] = RealClass(key=key, name=directory.name, directory=directory)

    if not classes:
        raise ValueError(f"No class folders found directly below: {real_root}")
    return classes


def _class_map_target(value: Any) -> str:
    """Extract a synset/class folder name from common JSON mapping formats."""
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, dict):
        for field in ("wnid", "synset", "class_name"):
            if field in value:
                return str(value[field])
    return str(value)


def _add_class_alias(
    aliases: dict[str, str],
    source: str,
    target: str,
    real_classes: Mapping[str, RealClass],
) -> None:
    source_key = canonical_class_key(source)
    target_key = canonical_class_key(target)
    if target_key not in real_classes:
        if source_key in real_classes:
            source_key, target_key = target_key, source_key
        else:
            return
    previous = aliases.get(source_key)
    if previous is not None and previous != target_key:
        raise ValueError(
            f"Class map assigns {source!r} to both {previous!r} and {target_key!r}"
        )
    aliases[source_key] = target_key


def load_class_aliases(
    class_map_path: Path,
    real_classes: Mapping[str, RealClass],
) -> dict[str, str]:
    """
    Load numeric-ID-to-folder aliases.

    JSON supports Keras ``imagenet_class_index.json``, simple index-to-synset
    objects, synset-to-index objects, or a list of synsets. Text files are
    interpreted as one synset per line in ImageNet class-index order.
    """
    if not class_map_path.is_file():
        raise ValueError(f"Class map file does not exist: {class_map_path}")

    aliases: dict[str, str] = {}
    if class_map_path.suffix.casefold() == ".json":
        data = json.loads(class_map_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            entries = enumerate(data)
        elif isinstance(data, dict):
            entries = data.items()
        else:
            raise ValueError("Class map JSON must be an object or list")
        for source, value in entries:
            _add_class_alias(
                aliases,
                str(source),
                _class_map_target(value),
                real_classes,
            )
    else:
        lines = [
            line.strip()
            for line in class_map_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for index, line in enumerate(lines):
            synset = line.split(maxsplit=1)[0]
            _add_class_alias(aliases, str(index), synset, real_classes)

    if not aliases:
        raise ValueError(
            f"Class map has no entries matching real folders below the dataset root: "
            f"{class_map_path}"
        )
    return aliases


def compile_class_regex(expression: str) -> re.Pattern[str]:
    """Compile and validate a user-provided class extraction expression."""
    try:
        pattern = re.compile(expression)
    except re.error as exc:
        raise ValueError(f"Invalid --class-regex: {exc}") from exc
    if "class_name" not in pattern.groupindex and pattern.groups < 1:
        raise ValueError(
            "--class-regex must contain a named group 'class_name' or at least one capture group"
        )
    return pattern


def captured_class_name(match: re.Match[str]) -> str:
    if "class_name" in match.re.groupindex:
        return match.group("class_name")
    return match.group(1)


def _has_name_boundaries(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not text[start - 1].isalnum()
    after_ok = end == len(text) or not text[end].isalnum()
    return before_ok and after_ok


def _folder_name_matches(filename_stem: str, real_classes: Mapping[str, RealClass]) -> list[str]:
    """Find nonnumeric real folder names appearing as delimited filename tokens."""
    folded_stem = filename_stem.casefold()
    matches = []
    for key, real_class in real_classes.items():
        if real_class.name.isdecimal():
            continue
        folded_name = real_class.name.casefold()
        start = folded_stem.find(folded_name)
        while start >= 0:
            end = start + len(folded_name)
            if _has_name_boundaries(folded_stem, start, end):
                matches.append(key)
                break
            start = folded_stem.find(folded_name, start + 1)
    if not matches:
        return []
    longest_name_length = max(len(real_classes[key].name) for key in matches)
    return [
        key for key in matches if len(real_classes[key].name) == longest_name_length
    ]


def class_key_from_filename(
    path: Path,
    real_classes: Mapping[str, RealClass],
    class_pattern: Optional[re.Pattern[str]] = None,
    class_aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Extract a class key from a generated image filename."""
    aliases = class_aliases or {}
    if class_pattern is not None:
        match = class_pattern.search(path.name)
        if match is None:
            return None
        key = canonical_class_key(captured_class_name(match))
        return aliases.get(key, key)

    match = DEFAULT_CLASS_PATTERN.search(path.stem)
    pattern_key = None
    if match is not None:
        pattern_key = canonical_class_key(match.group("class_name"))
        pattern_key = aliases.get(pattern_key, pattern_key)
        if pattern_key in real_classes:
            return pattern_key

    folder_matches = _folder_name_matches(path.stem, real_classes)
    if len(folder_matches) > 1:
        names = [real_classes[key].name for key in folder_matches]
        raise ValueError(f"Ambiguous class names in sample filename {path.name}: {names}")
    if folder_matches:
        return folder_matches[0]
    return pattern_key


def group_sample_images(
    samples_root: Path,
    real_classes: Mapping[str, RealClass],
    class_pattern: Optional[re.Pattern[str]] = None,
    class_aliases: Optional[Mapping[str, str]] = None,
    ignore_unmatched: bool = False,
) -> dict[str, list[Path]]:
    """Group flat or nested generated images by class encoded in each filename."""
    if not samples_root.is_dir():
        raise ValueError(f"Generated samples directory does not exist: {samples_root}")
    sample_paths = iter_image_files(samples_root)
    if not sample_paths:
        raise ValueError(f"No generated images found below: {samples_root}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    unmatched = []
    unknown = []
    for path in sample_paths:
        key = class_key_from_filename(
            path,
            real_classes,
            class_pattern,
            class_aliases,
        )
        if key is None:
            unmatched.append(path.name)
        elif key not in real_classes:
            unknown.append((path.name, key))
        else:
            grouped[key].append(path)

    if unknown:
        examples = ", ".join(f"{name} -> {key}" for name, key in unknown[:5])
        raise ValueError(f"Sample filenames reference classes without real folders: {examples}")
    if unmatched and not ignore_unmatched:
        examples = ", ".join(unmatched[:5])
        raise ValueError(
            "Could not extract classes from some sample filenames. "
            f"Examples: {examples}. Use --class-regex or --ignore-unmatched."
        )
    if not grouped:
        raise ValueError("No generated images could be matched to real class folders")
    return dict(grouped)


def select_class_keys(
    real_classes: Mapping[str, RealClass],
    grouped_samples: Mapping[str, Sequence[Path]],
    requested_classes: Optional[Iterable[str]],
    class_aliases: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Select requested classes or all classes represented by generated samples."""
    if requested_classes is None:
        return sorted(grouped_samples, key=lambda key: real_classes[key].name)

    selected = []
    aliases = class_aliases or {}
    for requested in requested_classes:
        key = canonical_class_key(requested)
        key = aliases.get(key, key)
        if key not in real_classes:
            raise ValueError(f"Requested class has no real folder: {requested}")
        if key not in grouped_samples:
            raise ValueError(f"Requested class has no generated samples: {requested}")
        if key not in selected:
            selected.append(key)
    return selected


class CleanFIDBackend:
    """Reuse one clean-fid Inception extractor for all class comparisons."""

    def __init__(
        self,
        mode: str,
        device: str,
        num_workers: int,
        batch_size: int,
        verbose: bool,
        use_dataparallel: bool,
    ) -> None:
        try:
            import torch
            from cleanfid import fid
        except ImportError as exc:
            raise RuntimeError(
                "clean-fid is not installed. Install it with: pip install clean-fid"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is not available: {device}")

        self.fid = fid
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = fid.build_feature_extractor(
            mode,
            self.device,
            use_dataparallel=use_dataparallel and self.device.type == "cuda",
        )
        self.mode = mode

    def _features(self, paths: Sequence[Path], description: str):
        return self.fid.get_files_features(
            [str(path) for path in paths],
            self.model,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            device=self.device,
            mode=self.mode,
            description=description,
            verbose=self.verbose,
        )

    def compute(
        self,
        generated_paths: Sequence[Path],
        real_paths: Sequence[Path],
        class_name: str,
    ) -> float:
        generated_features = self._features(
            generated_paths, f"FID generated {class_name}: "
        )
        real_features = self._features(real_paths, f"FID real {class_name}: ")
        return float(self.fid.fid_from_feats(generated_features, real_features))


def compute_classwise_scores(
    real_classes: Mapping[str, RealClass],
    grouped_samples: Mapping[str, Sequence[Path]],
    class_keys: Sequence[str],
    compute_fid: Callable[[Sequence[Path], Sequence[Path], str], float],
) -> list[dict[str, object]]:
    """Compute one score per selected class."""
    results = []
    for key in class_keys:
        real_class = real_classes[key]
        generated_paths = sorted(grouped_samples[key])
        real_paths = iter_image_files(real_class.directory)
        if len(generated_paths) < 2:
            raise ValueError(
                f"Class {real_class.name} has {len(generated_paths)} generated image(s); "
                "FID requires at least 2"
            )
        if len(real_paths) < 2:
            raise ValueError(
                f"Class {real_class.name} has {len(real_paths)} real image(s); "
                "FID requires at least 2"
            )
        if min(len(generated_paths), len(real_paths)) < 50:
            print(
                f"Warning: class {real_class.name} has only "
                f"{len(generated_paths)} generated and {len(real_paths)} real images; "
                "class-wise FID will be noisy.",
                file=sys.stderr,
            )

        score = compute_fid(generated_paths, real_paths, real_class.name)
        results.append(
            {
                "class_name": real_class.name,
                "class_key": key,
                "generated_images": len(generated_paths),
                "real_images": len(real_paths),
                "fid": score,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute clean-fid per class when real images are stored in class "
            "folders and generated images encode their class in the filename."
        )
    )
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument(
        "--class-regex",
        help=(
            "Regex applied to each generated filename. It must capture the class "
            "using (?P<class_name>...) or its first capture group."
        ),
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help=(
            "Optional ImageNet index-to-synset JSON or text file. Use this when "
            "filenames contain numeric IDs but real folders are named n########."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional class folder names or numeric IDs to evaluate.",
    )
    parser.add_argument(
        "--ignore-unmatched",
        action="store_true",
        help="Ignore image files whose filenames do not contain a recognized class.",
    )
    parser.add_argument(
        "--mode",
        choices=("clean", "legacy_pytorch", "legacy_tensorflow"),
        default="clean",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-dataparallel", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    real_root = args.real_dir.expanduser().resolve()
    samples_root = args.samples_dir.expanduser().resolve()
    real_classes = discover_real_classes(real_root)
    class_pattern = compile_class_regex(args.class_regex) if args.class_regex else None
    class_aliases = (
        load_class_aliases(args.class_map.expanduser().resolve(), real_classes)
        if args.class_map
        else {}
    )
    grouped_samples = group_sample_images(
        samples_root,
        real_classes,
        class_pattern=class_pattern,
        class_aliases=class_aliases,
        ignore_unmatched=args.ignore_unmatched,
    )
    class_keys = select_class_keys(
        real_classes,
        grouped_samples,
        args.classes,
        class_aliases,
    )

    backend = CleanFIDBackend(
        mode=args.mode,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        use_dataparallel=not args.no_dataparallel,
    )
    results = compute_classwise_scores(
        real_classes,
        grouped_samples,
        class_keys,
        backend.compute,
    )
    payload: dict[str, object] = {
        "real_dir": str(real_root),
        "samples_dir": str(samples_root),
        "mode": args.mode,
        "device": str(backend.device),
        "class_regex": args.class_regex,
        "class_map": str(args.class_map.expanduser().resolve()) if args.class_map else None,
        "mean_class_fid": fmean(float(result["fid"]) for result in results),
        "classes": results,
    }

    print(f"{'class':<24} {'generated':>10} {'real':>10} {'FID':>12}")
    for result in results:
        print(
            f"{str(result['class_name']):<24} "
            f"{int(result['generated_images']):>10} "
            f"{int(result['real_images']):>10} "
            f"{float(result['fid']):>12.4f}"
        )
    print(f"\nMean class FID: {float(payload['mean_class_fid']):.4f}")

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        print(f"Saved results to {output_path}")
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
