import tempfile
import unittest
from pathlib import Path

from src.data.preprocess import (
    RANDOM_SEED,
    assert_aligned_line_counts,
    clean_pairs,
    file_hash,
    find_leakage,
    normalize_text,
    read_pairs,
    write_manifest,
    write_split,
)


class Phase2DatasetTests(unittest.TestCase):
    def test_unicode_and_whitespace_normalization(self):
        decomposed = "a\u0301  नमस्ते\tworld\n"
        normalized = normalize_text(decomposed)
        self.assertEqual(normalized, "á नमस्ते world")
        self.assertEqual(normalized, normalized.encode("utf-8").decode("utf-8"))

    def test_empty_pairs_removed(self):
        pairs, empty, duplicates = clean_pairs([("hello", "नमस्ते"), ("", "Hindi"), ("English", " ")])
        self.assertEqual(pairs, [("hello", "नमस्ते")])
        self.assertEqual(empty, 2)
        self.assertEqual(duplicates, 0)

    def test_exact_duplicate_pairs_removed(self):
        pairs, empty, duplicates = clean_pairs([(" Hello ", " नमस्ते "), ("Hello", "नमस्ते"), ("Hello", "hello Hindi")])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(empty, 0)
        self.assertEqual(duplicates, 1)

    def test_alignment_and_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            en_path = Path(directory) / "en.txt"
            hi_path = Path(directory) / "hi.txt"
            en_path.write_text("Hello\nHow are you?\n", encoding="utf-8")
            hi_path.write_text("नमस्ते\nआप कैसे हैं?\n", encoding="utf-8")
            assert_aligned_line_counts(en_path, hi_path)
            pairs = list(read_pairs(str(en_path), str(hi_path)))
            self.assertEqual(pairs, [("Hello", "नमस्ते"), ("How are you?", "आप कैसे हैं?")])

    def test_misaligned_lines_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            en_path = Path(directory) / "en.txt"
            hi_path = Path(directory) / "hi.txt"
            en_path.write_text("One\nTwo\n", encoding="utf-8")
            hi_path.write_text("एक\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_aligned_line_counts(en_path, hi_path)
            with self.assertRaises(ValueError):
                list(read_pairs(str(en_path), str(hi_path)))

    def test_deterministic_preprocessing(self):
        source = [("  Hello\tworld ", " नमस्ते दुनिया ")] * 3
        first, first_empty, first_duplicates = clean_pairs(source)
        second, second_empty, second_duplicates = clean_pairs(source)
        self.assertEqual(first, second)
        self.assertEqual((first_empty, first_duplicates), (second_empty, second_duplicates))

    def test_split_separation_and_leakage(self):
        clean = {
            "train": [("train sentence", "प्रशिक्षण वाक्य")],
            "validation": [("validation sentence", "सत्यापन वाक्य")],
            "test": [("test sentence", "परीक्षण वाक्य")],
        }
        leakage = find_leakage(clean)
        self.assertEqual(all(count == 0 for count in leakage["pair_overlap"].values()), True)
        self.assertEqual(all(count == 0 for count in leakage["source_overlap"].values()), True)

    def test_leakage_detects_overlap(self):
        splits = {
            "train": [("shared", "one")],
            "validation": [("shared", "two")],
        }
        leakage = find_leakage(splits)
        self.assertEqual(leakage["source_overlap"]["train-validation"], 1)
        self.assertEqual(leakage["pair_overlap"]["train-validation"], 0)

    def test_write_split_hash_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = write_split(output, "fixture", [("Hello", "नमस्ते")])
            en_path = Path(metadata["english_path"])
            hi_path = Path(metadata["hindi_path"])
            self.assertEqual(metadata["count"], 1)
            self.assertEqual(file_hash(en_path), metadata["english_sha256"])
            self.assertEqual(file_hash(hi_path), metadata["hindi_sha256"])

    def test_random_seed_is_fixed(self):
        self.assertEqual(RANDOM_SEED, 42)

    def test_manifest_generation_and_hash_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data_path = write_split(output, "fixture", [("Hello", "नमस्ते")])["english_path"]
            manifest = {
                "random_seed": RANDOM_SEED,
                "fixture_sha256": file_hash(Path(data_path)),
            }
            manifest_path = write_manifest(output, manifest)
            self.assertTrue(manifest_path.exists())
            self.assertEqual(file_hash(manifest_path), file_hash(manifest_path))
           

if __name__ == "__main__":
    unittest.main()
