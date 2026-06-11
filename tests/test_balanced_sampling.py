import argparse
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from sample_balanced_ddp import (
    SampleRecord,
    assert_output_available,
    build_sample_records,
    create_labeled_npz,
    indices_for_rank,
    parse_sample_filename,
    run_sampling,
    sample_filename,
    validate_balanced_request,
    validate_generated_samples,
    write_manifest,
    write_metadata,
)


class BalancedSamplingTest(unittest.TestCase):
    def test_full_imagenet_50k_is_exactly_balanced(self):
        classes = list(range(1000))
        records = build_sample_records(50_000, classes)

        counts = Counter(record.class_id for record in records)
        self.assertEqual(len(records), 50_000)
        self.assertEqual(len(counts), 1000)
        self.assertEqual(set(counts.values()), {50})
        self.assertEqual(records[0].filename, "000000-class0000.png")
        self.assertEqual(records[1000].filename, "001000-class0000.png")

    def test_subset_balance_and_non_divisible_rejection(self):
        classes = [972, 973, 974, 975, 976]
        records = build_sample_records(50, classes)
        self.assertEqual(Counter(record.class_id for record in records), Counter({cls: 10 for cls in classes}))

        with self.assertRaisesRegex(ValueError, "not divisible"):
            validate_balanced_request(51, classes)

    def test_non_divisible_cli_request_fails_before_runtime_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"validation-only")
            args = argparse.Namespace(
                model="DiT-XL/2",
                ckpt=str(checkpoint),
                vae="mse",
                sample_dir=str(Path(directory) / "samples"),
                per_proc_batch_size=1,
                num_samples=51,
                image_size=256,
                num_classes=1000,
                classes=[972, 973, 974, 975, 976],
                cfg_scale=1.0,
                num_sampling_steps=250,
                global_seed=0,
                tf32=True,
            )

            with self.assertRaisesRegex(ValueError, "not divisible"):
                run_sampling(args)

    def test_rank_partition_has_no_padding_duplicates_or_omissions(self):
        num_samples = 50
        for world_size in (1, 2, 3, 4, 8, 64):
            partitions = [
                indices_for_rank(num_samples, rank, world_size)
                for rank in range(world_size)
            ]
            flattened = [index for partition in partitions for index in partition]
            self.assertEqual(sorted(flattened), list(range(num_samples)))
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_flat_artifacts_preserve_image_label_order(self):
        classes = [3, 7]
        records = build_sample_records(4, classes)

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            expected_pixels = []
            for record in records:
                pixel_value = record.index * 20
                image = np.full((2, 3, 3), pixel_value, dtype=np.uint8)
                Image.fromarray(image).save(run_dir / record.filename)
                expected_pixels.append(pixel_value)

            histogram = validate_generated_samples(run_dir, records)
            self.assertEqual(histogram, {3: 2, 7: 2})

            manifest_path = write_manifest(run_dir, records)
            with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
                rows = list(csv.DictReader(manifest_file))
            self.assertEqual([int(row["index"]) for row in rows], [0, 1, 2, 3])
            self.assertEqual([int(row["class_id"]) for row in rows], [3, 7, 3, 7])

            npz_path = create_labeled_npz(run_dir, records)
            with np.load(npz_path) as archive:
                self.assertEqual(archive["arr_0"].shape, (4, 2, 3, 3))
                self.assertEqual(archive["arr_0"].dtype, np.uint8)
                self.assertEqual(archive["arr_1"].dtype, np.int64)
                self.assertEqual(archive["arr_1"].tolist(), [3, 7, 3, 7])
                self.assertEqual(archive["arr_0"][:, 0, 0, 0].tolist(), expected_pixels)

            args = argparse.Namespace(
                model="DiT-XL/2",
                ckpt=str(Path(directory) / "checkpoint.pt"),
                image_size=256,
                vae="mse",
                num_classes=1000,
                num_samples=4,
                num_sampling_steps=250,
                cfg_scale=1.0,
                global_seed=0,
                tf32=True,
                per_proc_batch_size=2,
            )
            metadata_path = write_metadata(
                run_dir,
                args,
                classes,
                histogram,
                world_size=2,
                npz_path=npz_path,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["samples_per_class"], 2)
            self.assertEqual(metadata["class_counts"], {"3": 2, "7": 2})
            self.assertEqual(metadata["npz_arrays"]["arr_1"], "int64 class IDs aligned with arr_0")

    def test_filename_round_trip_and_stale_output_rejection(self):
        filename = sample_filename(42, 976)
        self.assertEqual(filename, "000042-class0976.png")
        self.assertEqual(
            parse_sample_filename(filename),
            SampleRecord(index=42, filename=filename, class_id=976),
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            npz_path = Path(f"{run_dir}.npz")
            run_dir.mkdir()
            assert_output_available(run_dir, npz_path)
            (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                assert_output_available(run_dir, npz_path)


if __name__ == "__main__":
    unittest.main()
