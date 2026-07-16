import tempfile
import unittest
from pathlib import Path

from compute_classwise_fid import (
    class_key_from_filename,
    compile_class_regex,
    compute_classwise_scores,
    discover_real_classes,
    group_sample_images,
    load_class_aliases,
    select_class_keys,
)


def touch_images(directory: Path, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"test")


class ClasswiseFIDTest(unittest.TestCase):
    def test_sampler_class_ids_match_zero_padded_real_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "00972", ["a.png", "b.png"])
            touch_images(root / "real" / "00973", ["a.png", "b.png"])
            touch_images(
                root / "samples",
                [
                    "000000-class0972.png",
                    "000001-class0973.png",
                    "000002-class0972.png",
                    "000003-class0973.png",
                ],
            )

            real_classes = discover_real_classes(root / "real")
            grouped = group_sample_images(root / "samples", real_classes)

            self.assertEqual(sorted(real_classes), ["972", "973"])
            self.assertEqual(len(grouped["972"]), 2)
            self.assertEqual(len(grouped["973"]), 2)

    def test_named_folder_is_found_in_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "retriever", ["a.jpg", "b.jpg"])
            touch_images(root / "real" / "golden_retriever", ["a.jpg", "b.jpg"])
            real_classes = discover_real_classes(root / "real")

            key = class_key_from_filename(
                Path("sample_class_golden_retriever_0001.png"),
                real_classes,
            )

            self.assertEqual(key, "golden_retriever")

    def test_imagenet_synset_in_filename_matches_real_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "n02099601", ["a.jpg", "b.jpg"])
            touch_images(
                root / "samples",
                ["000-classn02099601.png", "001-n02099601.png"],
            )
            real_classes = discover_real_classes(root / "real")

            grouped = group_sample_images(root / "samples", real_classes)

            self.assertEqual(len(grouped["n02099601"]), 2)

    def test_imagenet_numeric_id_maps_to_synset_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "n01440764", ["a.jpg", "b.jpg"])
            touch_images(root / "real" / "n01443537", ["a.jpg", "b.jpg"])
            touch_images(
                root / "samples",
                [
                    "000-class0000.png",
                    "001-class0001.png",
                    "002-class0000.png",
                    "003-class0001.png",
                ],
            )
            class_map_path = root / "imagenet_class_index.json"
            class_map_path.write_text(
                (
                    '{"0": ["n01440764", "tench"], '
                    '"1": ["n01443537", "goldfish"], '
                    '"2": ["n01484850", "great white shark"]}'
                ),
                encoding="utf-8",
            )
            real_classes = discover_real_classes(root / "real")
            aliases = load_class_aliases(class_map_path, real_classes)

            grouped = group_sample_images(
                root / "samples",
                real_classes,
                class_aliases=aliases,
            )
            selected = select_class_keys(
                real_classes,
                grouped,
                ["0"],
                aliases,
            )

            self.assertEqual(aliases, {"0": "n01440764", "1": "n01443537"})
            self.assertEqual(len(grouped["n01440764"]), 2)
            self.assertEqual(selected, ["n01440764"])

    def test_custom_regex_uses_named_or_first_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "cat", ["a.png", "b.png"])
            real_classes = discover_real_classes(root / "real")

            named = compile_class_regex(r"label=(?P<class_name>[a-z]+)")
            positional = compile_class_regex(r"target-([a-z]+)")

            self.assertEqual(
                class_key_from_filename(Path("x_label=cat_1.png"), real_classes, named),
                "cat",
            )
            self.assertEqual(
                class_key_from_filename(Path("x_target-cat_1.png"), real_classes, positional),
                "cat",
            )

    def test_unmatched_and_unknown_classes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "cat", ["a.png", "b.png"])
            real_classes = discover_real_classes(root / "real")

            touch_images(root / "unmatched", ["sample.png"])
            with self.assertRaisesRegex(ValueError, "Could not extract"):
                group_sample_images(root / "unmatched", real_classes)

            touch_images(root / "unknown", ["sample-classdog.png"])
            with self.assertRaisesRegex(ValueError, "without real folders"):
                group_sample_images(root / "unknown", real_classes)

    def test_score_orchestration_counts_images_and_selects_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            touch_images(root / "real" / "cat", ["a.png", "b.png"])
            touch_images(root / "real" / "dog", ["a.png", "b.png", "c.png"])
            touch_images(
                root / "samples",
                [
                    "000-classcat.png",
                    "001-classcat.png",
                    "002-classdog.png",
                    "003-classdog.png",
                ],
            )
            real_classes = discover_real_classes(root / "real")
            grouped = group_sample_images(root / "samples", real_classes)
            selected = select_class_keys(real_classes, grouped, ["dog"])
            calls = []

            def fake_fid(generated_paths, real_paths, class_name):
                calls.append((len(generated_paths), len(real_paths), class_name))
                return 12.5

            results = compute_classwise_scores(
                real_classes,
                grouped,
                selected,
                fake_fid,
            )

            self.assertEqual(calls, [(2, 3, "dog")])
            self.assertEqual(results[0]["generated_images"], 2)
            self.assertEqual(results[0]["real_images"], 3)
            self.assertEqual(results[0]["fid"], 12.5)


if __name__ == "__main__":
    unittest.main()
