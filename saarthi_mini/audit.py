"""Check the prepared split boundaries and preregistered challenge sources."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .data import ARCHIVE_SHA, VERSION, canonical, families, filter_reason, read_rows, sha, write_json


def audit(data, challenge_path):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["version"], manifest["source_sha256"]) != (VERSION, ARCHIVE_SHA):
        raise ValueError("Unexpected corpus version")
    rows, ids, keys, known_families = [], {}, set(), {}
    summary = {}
    for split in ("train", "validation", "test"):
        path = data / f"{split}.jsonl"
        expected = manifest["splits"][split]
        if sha(path) != expected["sha256"]:
            raise ValueError(f"Changed {split} file")
        loaded = read_rows(path)
        for row in loaded:
            key = canonical(row["en"]), canonical(row["hi"])
            if key in keys or row["id"] in ids or filter_reason(row["en"], row["hi"]):
                raise ValueError(f"Duplicate or filtered row in {split}: {row['id']}")
            if known_families.setdefault(row["family"], split) != split:
                raise ValueError(f"Stored family crosses splits: {row['family']}")
            keys.add(key)
            ids[row["id"]] = split, row
            rows.append(dict(row, split=split))
        if len(loaded) != expected["pairs"]:
            raise ValueError(f"Wrong row count in {split}")
        summary[split] = {
            "distinct_sources": len({canonical(r["en"]) for r in loaded}),
            "english_word_lengths": dict(sorted(Counter(len(canonical(r["en"]).split())
                                                         for r in loaded).items())),
        }
    for group in families(rows):
        if len({row["split"] for row in group}) != 1:
            raise ValueError("Recomputed family crosses corpus splits")
    probes = json.loads(challenge_path.read_text(encoding="utf-8"))["probes"]
    synthetic_sources = []
    for probe in probes:
        if len(probe["variants"]) != 2 or len(probe["facts"]) != 2:
            raise ValueError(f"Expected two sides for {probe['id']}")
        for i, key in enumerate(("anchor", "companion")):
            if key in probe:
                partition, row = ids[probe[key]]
                if partition != "test" or canonical(probe["variants"][i]) != canonical(row["en"]):
                    raise ValueError(f"Wrong source anchor for {probe['id']}")
        synthetic_sources.extend({"split": "test", "en": text, "id": probe["id"]}
                                 for text in probe["variants"])
    for group in families(rows + synthetic_sources, languages=("en",)):
        split_names = {row["split"] for row in group}
        if "test" in split_names and split_names != {"test"}:
            raise ValueError(f"Challenge family crosses splits: {[r['id'] for r in group]}")
    summary["probes"] = dict(Counter(probe["kind"] for probe in probes))
    summary["cross_split_families_detected"] = 0
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--challenges", type=Path, default=Path(__file__).with_name("challenges.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.data, args.challenges)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
