#!/usr/bin/env python3
"""Create a row-aligned comparison grid for DiT, DuoDiT, and LightningDiT."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MODEL_SPECS = (
    ("DiT", "dit", "{index:06d}-class{index:04d}.png"),
    ("DuoDiT", "duodit", "{index:06d}-class{index:04d}.png"),
    ("LightningDiT", "light", "{index:06d}.png"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("visuals/model_comparison_grid.png"),
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--header-height", type=int, default=64)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    indices = range(args.start_index, args.start_index + args.rows)

    paths = [
        [
            args.samples_root / directory / pattern.format(index=index)
            for _, directory, pattern in MODEL_SPECS
        ]
        for index in indices
    ]
    missing = [path for row in paths for path in row if not path.is_file()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing comparison samples:\n{missing_list}")

    with Image.open(paths[0][0]) as first:
        cell_size = first.size

    width = len(MODEL_SPECS) * cell_size[0] + (len(MODEL_SPECS) - 1) * args.gap
    grid_height = args.rows * cell_size[1] + (args.rows - 1) * args.gap
    height = args.header_height + grid_height

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(28)

    for column, (label, _, _) in enumerate(MODEL_SPECS):
        x = column * (cell_size[0] + args.gap)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (
                x + (cell_size[0] - text_width) / 2,
                (args.header_height - text_height) / 2 - box[1],
            ),
            label,
            fill="black",
            font=font,
        )

    for row, row_paths in enumerate(paths):
        y = args.header_height + row * (cell_size[1] + args.gap)
        for column, path in enumerate(row_paths):
            x = column * (cell_size[0] + args.gap)
            with Image.open(path) as sample:
                image = sample.convert("RGB")
                if image.size != cell_size:
                    raise ValueError(
                        f"{path} has size {image.size}; expected {cell_size}"
                    )
                canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + cell_size[0] - 1, y + cell_size[1] - 1),
                outline=(210, 210, 210),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True)
    print(f"Saved {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
