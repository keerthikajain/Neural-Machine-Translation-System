"""Create unscored candidate and development-diagnostic review material."""
import argparse
from collections import Counter
import csv
from pathlib import Path

from .data import sha, write_json, write_rows
from .pilot4_curate import CHECKS, dataset, diagnostic_issues, identity, metadata


def candidate_sheet(path, manifest, flags):
    ids = {r["candidate"]["id"] for r in manifest["rows"]}
    if not flags.keys() <= ids:
        raise ValueError("Analyst flag refers to an unknown candidate")
    fields = ("id", "slot", "category", "structure", "en", "hi", "pair_sha256", "source",
              "license_status", "automated_overlap", "analyst_note", "reviewer", "english_grammar",
              "hindi_grammar", *CHECKS, "ambiguity_or_register", "review_decision", "proposed_edit")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest["rows"]:
            candidate = row["candidate"]
            writer.writerow({
                **{key: candidate[key] for key in ("id", "category", "structure", "en", "hi")},
                "slot": row["position_zero_based"], "pair_sha256": row["pair_sha256"],
                "source": row["provenance"]["source_reference"],
                "license_status": row["provenance"]["license"],
                "automated_overlap": str(row["overlap"]),
                "analyst_note": flags.get(candidate["id"], "No specific analyst flag. Independent review required."),
            })


def prepare(args):
    _, splits, tokenizer = dataset(args)
    manifest = metadata(args.manifest)
    if manifest["parent_manifest_sha256"] != sha(args.data / "manifest.json"):
        raise ValueError("Candidate manifest refers to a different parent")
    if len(manifest["rows"]) != 120 or any(
            r["status"] != "pending" or r["approvals"] or identity(r["candidate"]) != r["pair_sha256"]
            for r in manifest["rows"]):
        raise ValueError("Expected the original pending cohort and pair digests")
    with args.draft.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row, id=f"dev04-{index:02}", tags=row["tags"].split(","),
                     approval_status="pending")
                for index, row in enumerate(csv.DictReader(handle, delimiter="\t"), 1)]
    issues, _ = diagnostic_issues(splits["train"], splits["validation"],
                                  [r["candidate"] for r in manifest["rows"]], rows, tokenizer)
    args.output.mkdir()
    write_rows(args.output / "diagnostic-draft.jsonl", rows)
    candidate_sheet(args.output / "candidate-review.csv", manifest, metadata(args.flags))
    write_json(args.output / "screen.json", {
        "status": "pending", "diagnostic_rows": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "tag_counts": dict(Counter(tag for row in rows for tag in row["tags"])),
        "diagnostic_sha256": sha(args.output / "diagnostic-draft.jsonl"),
        "candidate_manifest_sha256": sha(args.manifest), "development_issues": issues,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "tokenizer", "manifest", "draft", "flags", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    prepare(p.parse_args())
