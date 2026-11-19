import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import sentencepiece as spm

DATASET_ID = "cfilt/iitb-english-hindi"
DATASET_CONFIG = "default"
RANDOM_SEED = 42


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def clean_pairs(pairs: Iterable[Tuple[str, str]]):
    """Normalize pairs, remove empty pairs, and remove exact normalized duplicates."""
    seen = set()
    kept = []
    empty_removed = 0
    duplicates_removed = 0

    for raw_en, raw_hi in pairs:
        en = normalize_text(raw_en)
        hi = normalize_text(raw_hi)
        if not en or not hi:
            empty_removed += 1
            continue
        pair = (en, hi)
        if pair in seen:
            duplicates_removed += 1
            continue
        seen.add(pair)
        kept.append(pair)

    return kept, empty_removed, duplicates_removed


def read_pairs(en_path: str, hi_path: str) -> Iterator[Tuple[str, str]]:
    en_file_path = Path(en_path)
    hi_file_path = Path(hi_path)
    assert_aligned_line_counts(en_file_path, hi_file_path)

    with open(en_file_path, "r", encoding="utf-8") as en_file, open(hi_file_path, "r", encoding="utf-8") as hi_file:
        for en_line, hi_line in zip(en_file, hi_file):
            en = normalize_text(en_line)
            hi = normalize_text(hi_line)
            if en and hi:
                yield (en, hi)


def assert_aligned_line_counts(en_path: Path, hi_path: Path) -> None:
    with open(en_path, "r", encoding="utf-8") as en_file:
        en_count = sum(1 for _ in en_file)
    with open(hi_path, "r", encoding="utf-8") as hi_file:
        hi_count = sum(1 for _ in hi_file)
    if en_count != hi_count:
        raise ValueError(f"English/Hindi line mismatch: {en_count} vs {hi_count}")


def find_leakage(splits: dict) -> dict:
    pairs = {name: set(split_pairs) for name, split_pairs in splits.items()}
    sources = {name: {en for en, _ in split_pairs} for name, split_pairs in splits.items()}
    names = list(splits.keys())
    pair_overlap = {}
    source_overlap = {}

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}-{right}"
            pair_overlap[key] = len(pairs[left].intersection(pairs[right]))
            source_overlap[key] = len(sources[left].intersection(sources[right]))

    return {"pair_overlap": pair_overlap, "source_overlap": source_overlap}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(output_dir: Path, split_name: str, pairs: Iterable[Tuple[str, str]]) -> dict:
    en_path = output_dir / f"{split_name}_en.txt"
    hi_path = output_dir / f"{split_name}_hi.txt"
    count = 0

    with open(en_path, "w", encoding="utf-8", newline="\n") as en_file, open(hi_path, "w", encoding="utf-8", newline="\n") as hi_file:
        for en, hi in pairs:
            en_file.write(en + "\n")
            hi_file.write(hi + "\n")
            count += 1

    assert_aligned_line_counts(en_path, hi_path)
    return {
        "count": count,
        "english_path": en_path.as_posix(),
        "hindi_path": hi_path.as_posix(),
        "english_sha256": file_hash(en_path),
        "hindi_sha256": file_hash(hi_path),
    }


def percentile(sorted_values: list, percentile_value: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def sequence_statistics(pairs: Iterable[Tuple[str, str]], tokenizer_path: Path) -> dict:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    english_lengths = []
    hindi_lengths = []

    for en, hi in pairs:
        english_lengths.append(len(processor.encode_as_ids(en)))
        hindi_lengths.append(len(processor.encode_as_ids(hi)))

    def describe(lengths: list) -> dict:
        values = sorted(lengths)
        count = len(values)

        def count_exceeding(limit: int):
            number = sum(length > limit for length in values)
            percent = round(100.0 * number / count, 4) if count else 0.0
            return number, percent

        exceeding_128, percent_128 = count_exceeding(128)
        exceeding_256, percent_256 = count_exceeding(256)
        return {
            "pair_count": count,
            "mean_tokens": round(sum(values) / count, 2) if count else 0.0,
            "median_tokens": round(percentile(values, 0.50), 2) if count else 0.0,
            "p95_tokens": round(percentile(values, 0.95), 2) if count else 0.0,
            "p99_tokens": round(percentile(values, 0.99), 2) if count else 0.0,
            "maximum_tokens": max(values) if values else 0,
            "exceeding_128": exceeding_128,
            "exceeding_128_percent": percent_128,
            "exceeding_256": exceeding_256,
            "exceeding_256_percent": percent_256,
        }

    return {"english": describe(english_lengths), "hindi": describe(hindi_lengths)}


def dataset_metadata():
    from datasets import load_dataset_builder
    builder = load_dataset_builder(DATASET_ID, name=DATASET_CONFIG)
    version = getattr(builder.info, "version", None)
    return {
        "identifier": DATASET_ID,
        "configuration": DATASET_CONFIG,
        "builder_version": str(version) if version is not None else None,
        "release_version_verified": False,
    }


def local_manifest(output_dir: Path, tokenizer_path: Path):
    random.seed(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_files = {
        "train": (Path("data/processed/clean_train_en.txt"), Path("data/processed/clean_train_hi.txt")),
        "validation": (Path("data/processed/val_en.txt"), Path("data/processed/val_hi.txt")),
    }
    cleaned_splits = {}
    removal_counts = {}
    files = {}

    for split_name, (en_path, hi_path) in local_files.items():
        if not en_path.exists() or not hi_path.exists():
            continue
        assert_aligned_line_counts(en_path, hi_path)
        pairs = list(read_pairs(str(en_path), str(hi_path)))
        cleaned_splits[split_name] = pairs
        removal_counts[split_name] = {"empty_pairs_removed": None, "duplicate_pairs_removed": None, "note": "Requires rebuilding from the Hugging Face source split."}
        files[split_name] = {
            "count": len(pairs),
            "english_path": en_path.as_posix(),
            "hindi_path": hi_path.as_posix(),
            "english_sha256": file_hash(en_path),
            "hindi_sha256": file_hash(hi_path),
        }

    if "train" not in cleaned_splits:
        raise FileNotFoundError("Existing cleaned training files were not found")

    sequence_lengths = {
        name: sequence_statistics(pairs, tokenizer_path)
        for name, pairs in cleaned_splits.items()
    }
    leakage = find_leakage(cleaned_splits)

    manifest = {
        "dataset": dataset_metadata(),
        "random_seed": RANDOM_SEED,
        "original_split_counts": {"train": 1659083, "validation": 520, "test": 2507},
        "cleaned_split_counts": {name: len(pairs) for name, pairs in cleaned_splits.items()},
        "removal_counts": removal_counts,
        "normalization_rules": {
            "unicode": "NFC",
            "whitespace": "collapse whitespace runs and trim",
            "empty_pairs_removed": True,
            "duplicate_method": "exact normalized (English, Hindi) pair",
            "punctuation": "preserved",
        },
        "split_policy": "Existing original train and validation files only; no custom shuffling or split creation.",
        "files": files,
        "leakage": leakage,
        "sequence_lengths": sequence_lengths,
        "tokenizer_path": tokenizer_path.as_posix(),
        "available_local_splits": list(cleaned_splits.keys()),
    }

    manifest_path = write_manifest(output_dir, manifest)
    return manifest, manifest_path


def full_manifest(output_dir: Path, tokenizer_path: Path):
    from datasets import load_dataset
    random.seed(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID)

    cleaned_splits = {}
    original_counts = {}
    removal_counts = {}
    files = {}

    for split_name in ("train", "validation", "test"):
        original_name = "validation" if split_name == "validation" else split_name
        original_counts[split_name] = len(dataset[original_name])
        raw_pairs = ((example["translation"]["en"], example["translation"]["hi"])
                     for example in dataset[original_name])
        pairs, empty_removed, duplicates_removed = clean_pairs(raw_pairs)
        cleaned_splits[split_name] = pairs
        removal_counts[split_name] = {
            "empty_pairs_removed": empty_removed,
            "duplicate_pairs_removed": duplicates_removed,
        }
        files[split_name] = write_split(output_dir, split_name, pairs)

    sequence_lengths = {
        name: sequence_statistics(pairs, tokenizer_path)
        for name, pairs in cleaned_splits.items()
    }
    leakage = find_leakage(cleaned_splits)

    manifest = {
        "dataset": dataset_metadata(),
        "random_seed": RANDOM_SEED,
        "original_split_counts": original_counts,
        "cleaned_split_counts": {name: len(pairs) for name, pairs in cleaned_splits.items()},
        "removal_counts": removal_counts,
        "normalization_rules": {
            "unicode": "NFC",
            "whitespace": "collapse whitespace runs and trim",
            "empty_pairs_removed": True,
            "duplicate_method": "exact normalized (English, Hindi) pair",
            "punctuation": "preserved",
        },
        "split_policy": "Original Hugging Face train, validation, and test splits; no custom shuffling or split creation.",
        "files": files,
        "leakage": leakage,
        "sequence_lengths": sequence_lengths,
        "tokenizer_path": tokenizer_path.as_posix(),
        "available_local_splits": ["train", "validation", "test"],
    }

    manifest_path = write_manifest(output_dir, manifest)
    return manifest, manifest_path


def write_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Build a reproducible IITB dataset export and manifest")
    parser.add_argument("--output-dir", default="data/phase2")
    parser.add_argument("--tokenizer", default="data/processed/tokenizer/sp_model.model")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer)
    manifest, manifest_path = local_manifest(output_dir, tokenizer_path) if args.local_only else full_manifest(output_dir, tokenizer_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
