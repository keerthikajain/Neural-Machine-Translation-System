"""Prepare pending development candidates or build a separately approved corpus."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import ZipFile

import sentencepiece as spm

from .data import canonical, families, filter_reason, read_rows, sha, write_json, write_rows
from .pilot4_audit import NAME, NEGATIVE, QUANTITY, REQUEST, split_payload
from .train import load_verified

BASE = "mini-tatoeba-20260213-v1"
TOKENIZER = "7d3bf82ecc9da10e0e51cd89c0685be09cfd9007936a312dbefb93360154a0f0"
LANGUAGES = ("en", "hi")
CHECKS = ("meaning", "action_roles", "quantity_referent", "negation", "naturalness")


def metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lengths(row, tokenizer):
    return tuple(len(tokenizer.encode(row[lang])) for lang in LANGUAGES)


def identity(row):
    text = json.dumps({k: row[k] for k in LANGUAGES}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def overlap(train, validation, candidates, diagnostic=()):
    """Run the existing connected-family algorithm over development material only."""
    rows = [dict(row, partition=name) for name, items in (
        ("train", train), ("validation", validation), ("candidate", candidates),
        ("diagnostic", diagnostic)) for row in items]
    flagged = {}
    for family in families(rows):
        sources = Counter(row["partition"] for row in family)
        for row in family:
            if row["partition"] == "candidate":
                flagged[row["id"]] = {"validation_family_rows": sources["validation"],
                                      "diagnostic_family_rows": sources["diagnostic"],
                                      "candidate_family_rows": sources["candidate"],
                                      "train_family_rows": sources["train"]}
            if row["partition"] == "diagnostic":
                flagged[row["id"]] = {"train_or_candidate_family_rows":
                                       sources["train"] + sources["candidate"],
                                       "validation_family_rows": sources["validation"]}
    return flagged


def dataset(args):
    m = metadata(args.data / "manifest.json")
    if m["version"] != BASE or sha(args.tokenizer) != TOKENIZER:
        raise ValueError("Original manifest version or tokenizer digest mismatch")
    return m, load_verified(args.data, ("train", "validation")), spm.SentencePieceProcessor(
        model_file=str(args.tokenizer))


def quality(row, tokenizer):
    reason = filter_reason(row["en"], row["hi"])
    if reason:
        return reason
    if max(lengths(row, tokenizer)) > 62:
        return "token_limit"
    return None


def feature_summary(rows, tokenizer):
    result = {}
    for lang, lengths_by_row in zip(LANGUAGES, zip(*(lengths(r, tokenizer) for r in rows)), strict=True):
        words = [w for row in rows for w in canonical(row[lang]).split()]
        result[lang] = {"surface_types": len(set(words)), "token_length_mean":
                        sum(lengths_by_row) / len(rows), "token_length_max": max(lengths_by_row)}
    for name, predicate in {
        "question": lambda s: s.endswith("?"), "negation_cue": NEGATIVE.search,
        "request_cue": REQUEST.search, "quantity_cue": QUANTITY.search, "tracked_name_cue": NAME.search,
    }.items():
        result[name] = sum(bool(predicate(row["en"].lower())) for row in rows)
    return result


def prepare(args):
    manifest, splits, tokenizer = dataset(args)
    train, validation = splits.values()
    with args.candidates.open(encoding="utf-8", newline="") as handle:
        drafts = list(csv.DictReader(handle, delimiter="\t"))
    if len(drafts) != 120 or Counter(r["category"] for r in drafts) != {"action": 60, "quantity": 60}:
        raise ValueError("Expected a 60/60 draft cohort")
    candidates = [dict(r, id=f"p4-curated-{i:03}") for i, r in enumerate(drafts, 1)]
    if len({identity(row) for row in candidates}) != len(candidates):
        raise ValueError("Repeated draft pair")
    counts = Counter(canonical(row["en"]) for row in train)
    slots = {i: row for i, row in enumerate(train) if counts[canonical(row["en"])] > 1}
    remaining = counts.copy()
    slot_lengths = {i: lengths(row, tokenizer) for i, row in slots.items()}
    def controls(row):
        en = row["en"]
        return en.endswith("?"), bool(NAME.search(en)), bool(NEGATIVE.search(en))

    slot_controls = {i: controls(row) for i, row in slots.items()}
    retained_families = Counter(row["family"] for row in train)
    overlaps = overlap(train, validation, candidates)
    old_vocabulary = {lang: {word for row in train for word in canonical(row[lang]).split()}
                      for lang in LANGUAGES}
    records = []
    for new in candidates:
        target = lengths(new, tokenizer)
        options = [index for index, row in slots.items() if remaining[canonical(row["en"])] > 1
                   and retained_families[row["family"]] > 1]
        if not options:
            raise ValueError("Insufficient replaceable alternate-reference slots")
        expected = controls(new)
        index = min(options, key=lambda i: (
            *(a != b for a, b in zip(slot_controls[i], expected, strict=True)),
            sum(abs(x - y) for x, y in zip(target, slot_lengths[i], strict=True)), i))
        old = slots.pop(index)
        remaining[canonical(old["en"])] -= 1
        retained_families[old["family"]] -= 1
        records.append({
            "position_zero_based": index, "original": old, "candidate": new,
            "pair_sha256": identity(new), "kind": "new_curated_example", "status": "pending",
            "reason": "Replace an alternate reference while retaining the source and expand bilingual "
                      + ("action–role coverage" if new["category"] == "action" else "quantity–referent coverage"),
            "provenance": {"origin": "Project-authored assistant draft, no external text imported",
                           "source_reference": f"candidate-pairs.tsv:{len(records) + 2}",
                           "license": "pending owner authorization", "source_file_sha256": sha(args.candidates)},
            "analyst_checks": {"source_target_screen": quality(new, tokenizer),
                               "original_piece_lengths": slot_lengths[index], "candidate_piece_lengths": target},
            "overlap": overlaps[new["id"]], "approvals": [],
        })
    projected = train.copy()
    for row in records:
        projected[row["position_zero_based"]] = row["candidate"]
    report = {
        "version": "mini-pilot4-curation-draft-v1", "parent_manifest_sha256": sha(args.data / "manifest.json"),
        "parent_train_sha256": manifest["splits"]["train"]["sha256"],
        "parent_validation_sha256": manifest["splits"]["validation"]["sha256"],
        "tokenizer_sha256": TOKENIZER, "status": "pending_review",
        "selection": "Train-only alternate references, question/name/negation cues, bilingual token-length distance, original position tie-break",
        "distribution": {"before": feature_summary(train, tokenizer),
                         "projected": feature_summary(projected, tokenizer),
                         "new_surface_word_forms": {lang: sum(
                             word not in old_vocabulary[lang]
                             for word in {w for row in candidates for w in canonical(row[lang]).split()})
                             for lang in LANGUAGES}},
        "rows": records,
    }
    write_json(args.output, report)


def approved(record):
    reviews = record["approvals"]
    identities = {r.get("reviewer", "").strip().casefold() for r in reviews}
    return (record["status"] == "approved" and len(reviews) == 2 and len(identities) == 2
            and "" not in identities and all(
                r.get("decision") == "approve" and r.get("pair_sha256") == record["pair_sha256"]
                and all(r.get("checks", {}).get(check) == "pass" for check in CHECKS)
                for r in reviews))


def diagnostic_issues(train, validation, candidates, diagnostic, tokenizer):
    """Check the frozen diagnostic policy using development files only."""
    issues = {}
    if (len(diagnostic) != 60
            or Counter(r["category"] for r in diagnostic) != {"action": 30, "quantity": 30}):
        issues["size_or_strata"] = True
    if len({r["id"] for r in diagnostic}) != len(diagnostic):
        issues["duplicate_ids"] = True
    if sum("negation" in r.get("tags", ()) for r in diagnostic) < 10:
        issues["negation_quota"] = True
    # Human review still verifies semantic tags, idioms, and paraphrases beyond this family check.
    quality_failures = {r["id"]: reason for r in diagnostic if (reason := quality(r, tokenizer))}
    if quality_failures:
        issues["surface_or_length"] = quality_failures
    duplicate_families = [[r["id"] for r in group] for group in families(diagnostic) if len(group) > 1]
    if duplicate_families:
        issues["diagnostic_families"] = duplicate_families
    hits = overlap(train, validation, candidates, diagnostic)
    conflicts = {r["id"]: hits[r["id"]] for r in diagnostic if any(hits[r["id"]].values())}
    if conflicts:
        issues["control_validation_or_candidate_overlap"] = conflicts
    return issues, hits


def build(args):
    ready = metadata(args.manifest)
    records = ready["rows"]
    if len(records) != 120 or not all(approved(row) for row in records):
        raise ValueError("Exactly 120 replacements need two distinct bilingual approvals")
    manifest, splits, tokenizer = dataset(args)
    if ready["parent_manifest_sha256"] != sha(args.data / "manifest.json") or ready["tokenizer_sha256"] != TOKENIZER:
        raise ValueError("Candidate manifest provenance mismatch")
    train, validation = splits.values()
    diagnostic = read_rows(args.diagnostic)
    if (len(diagnostic) != 60 or Counter(r["category"] for r in diagnostic) != {"action": 30, "quantity": 30}
            or len({r["id"] for r in diagnostic}) != 60):
        raise ValueError("Freeze a 60-item diagnostic with unique IDs and two 30-item strata")
    approvals = metadata(args.attestation)
    if (not approvals.get("steward", "").strip() or not approvals.get("method_or_record", "").strip()
            or approvals.get("candidate_manifest_sha256") != sha(args.manifest)
            or approvals.get("diagnostic_sha256") != sha(args.diagnostic)
            or any(approvals.get(key) != "pass" for key in
                   ("sealed_overlap", "diagnostic_bilingual_approval", "independent_30_item_review"))):
        raise ValueError("Missing or stale readiness attestation")
    positions = set()
    retained = train.copy()
    new_rows = []
    for record in records:
        i, new = record["position_zero_based"], record["candidate"]
        provenance = record["provenance"]
        if (not isinstance(i, int) or not 0 <= i < len(train) or i in positions
                or record["original"] != train[i] or identity(new) != record["pair_sha256"]
                or quality(new, tokenizer) or not provenance.get("license_approval")
                or not provenance.get("license") or "pending" in provenance["license"].casefold()):
            raise ValueError(f"Invalid replacement for slot {i}")
        positions.add(i)
        new_rows.append(dict(new, attribution=provenance))
    if Counter(r["category"] for r in new_rows) != {"action": 60, "quantity": 60}:
        raise ValueError("Replacement strata changed")
    old_rows = [row for i, row in enumerate(train) if i not in positions]
    if {canonical(r["en"]) for r in old_rows} != {canonical(r["en"]) for r in train}:
        raise ValueError("Replacement removes the last reference of an original source")
    combined_ids = [r["id"] for r in old_rows + validation + new_rows + diagnostic]
    if len(set(combined_ids)) != len(combined_ids):
        raise ValueError("Example IDs must be unique across development inputs")
    issues, hits = diagnostic_issues(train, validation, new_rows, diagnostic, tokenizer)
    if issues:
        raise ValueError(f"Diagnostic fails development checks: {', '.join(issues)}")
    for new in new_rows:
        hit = hits[new["id"]]
        if (hit["validation_family_rows"] or hit["diagnostic_family_rows"] or hit["train_family_rows"]
                or hit["candidate_family_rows"] != 1):
            raise ValueError(f"Candidate family exceeds overlap policy: {new['id']}")
    known = {lang: {canonical(r[lang]) for r in old_rows} for lang in LANGUAGES}
    for record, new in zip(records, new_rows, strict=True):
        if any(canonical(new[lang]) in known[lang] for lang in LANGUAGES):
            raise ValueError(f"Candidate duplicates retained text: {new['id']}")
        retained[record["position_zero_based"]] = dict(new, family=new["id"])
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        out = Path(staging) / "corpus"
        out.mkdir()
        write_rows(out / "train.jsonl", retained)
        shutil.copyfile(args.data / "validation.jsonl", out / "validation.jsonl")
        split = split_payload(retained)
        manifest["splits"]["train"] = {
            "pairs": len(retained), "families": split["families"], "questions": split["questions"],
            "sha256": sha(out / "train.jsonl"),
        }
        manifest["version"] = "mini-pilot4-approved-replacement-v1"
        manifest["sealed_test_external"] = True
        manifest["parent_manifest_sha256"] = ready["parent_manifest_sha256"]
        manifest["approved_manifest_sha256"] = sha(args.manifest)
        manifest["approval_attestation_sha256"] = sha(args.attestation)
        manifest["diagnostic_sha256"] = sha(args.diagnostic)
        manifest["source_partition"] = "original manifest retains attribution for unchanged rows"
        manifest["intervention"] = {"replacements": len(records),
                                    "actual_distribution": feature_summary(retained, tokenizer)}
        write_json(out / "manifest.json", manifest)
        out.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action, fields in (("prepare", ("candidates",)), ("build", ("manifest", "diagnostic", "attestation"))):
        command = sub.add_parser(action)
        for field in ("data", "tokenizer", "output", *fields):
            command.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else build)(args)


if __name__ == "__main__":
    main()
