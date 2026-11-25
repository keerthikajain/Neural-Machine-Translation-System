"""Quantify development-only coverage and verify the Pilot 4 annotation sample."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

from .data import canonical, families, sha, write_json
from .decode_diagnostics import length_stats
from .evaluate import fact_proxies
from .metrics import scores, unique_sources
from .train import load_verified

RESULT_SHA = "0a05d117eb8673f0907ff303bdb28ab46c018ce697e740605ae791f703acd6ec"
NAME = re.compile(r"\b(?:tom|mary|john|india|delhi)\b", re.I)
NEGATIVE = re.compile(r"\b(?:not|never|no|cannot|nothing|nobody|none|\w+n['’]t)\b", re.I)
QUANTITY = re.compile(r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                      r"twelve|thirteen|fourteen|fifteen|twenty|thirty|hundred|thousand|"
                      r"half|quarter|dozen|several|few|many|much)\b")
REQUEST = re.compile(r"^(?:please\b|(?:can|could|would|will) you\b|let's\b|do not\b|don't\b)")
ACTIONS = {
    "buy": r"buy|buys|bought|buying|purchase|purchased",
    "see": r"see|sees|saw|seen|seeing",
    "give": r"give|gives|gave|given|giving",
    "return": r"return|returns|returned|returning|give .* back|gave .* back",
    "speak": r"speak|speaks|spoke|spoken|speaking|talk|talks|talked|talking",
    "arrive": r"arrive|arrives|arrived|arriving|come back|comes back|came back",
    "die": r"die|dies|died|dying|passed away|pass away",
    "spoil": r"spoil|spoils|spoiled|spoilt|go bad|goes bad|went bad",
}
PROBES = [
    ("see_mouse", "see", r"mouse|mice", r"देख|नज़र", r"चूह[ाेों]"),
    ("see_dog", "see", r"dog|dogs", r"देख|नज़र", r"कुत्त[ाेों]"),
    ("buy_milk", "buy", r"milk", r"ख[़]?रीद", r"दूध"),
    ("buy_bottles", "buy", r"bottle|bottles", r"ख[़]?रीद", r"बोतल"),
    ("give_message", "give", r"message|messages|note|notes", r"देन|दिया|दिए|देने|देती|देगा|भेज", r"संदेश|नोट"),
    ("return_recipient", "return", r"(?:to|back to) (?:tom|mary|john|him|her|me|you|us|them)",
     r"लौटा|वापस", r"को|उसे|मुझे|तुम्हें|आपको|उन्हें"),
    ("speak_language", "speak", r"languages?|english|hindi|french|spanish", r"बोल|बात", r"भाषा|अंग्रेज़ी|अंग्रेजी|हिंदी|फ़्रेंच"),
    ("spoil_milk", "spoil", r"milk", r"खराब|ख़राब|बिगड़", r"दूध"),
    ("die_hospital", "die", r"hospital", r"चल बस|मर|गुज़र|निधन|मौत", r"अस्पताल|हस्पताल"),
    ("arrive_tom", "arrive", r"tom", r"आए|आय|पहुंच|पहुँच|वापस", r"टॉम|टोम|टाम"),
]
QUANTITY_PROBES = {
    "two_bottles": r"two bottles?", "two_murders": r"two murders?",
    "two_children": r"two (?:children|kids)", "number_children": r"(?:one|two|three|four|five|\d+) (?:children|kids)",
    "ten_minutes": r"ten minutes?", "clock_half": r"half past|eight thirty",
    "clock_hour": r"o'clock", "duration_minutes": r"(?:one|two|three|four|five|ten|\d+) minutes?",
    "duration_years": r"(?:one|two|three|four|five|ten|\d+) years?",
}


def match(expression, text):
    return re.search(rf"\b(?:{expression})\b", text) is not None


def sample_ids(train, validation, blind_ids):
    groups = {key: [] for key in ("negation", "question", "name", "other")}
    for row in validation:
        en, ident = row["en"], row["ids"][0]
        if ident in blind_ids:
            continue
        key = ("negation" if NEGATIVE.search(en) else "question" if en.endswith("?")
               else "name" if NAME.search(en) else "other")
        groups[key].append(ident)
    rng = random.Random(40511)
    return {"validation_seed": 40511, "validation_group_sizes": {k: len(v) for k, v in groups.items()},
            "validation_ids": {k: rng.sample(v, 6) for k, v in groups.items()}, "train_seed": 40512,
            "train_ids": [r["id"] for r in random.Random(40512).sample(train, 24)]}


def split_payload(rows):
    sources = unique_sources(rows)
    en = [canonical(row["en"]) for row in rows]
    actions = {name: sum(match(pattern, text) for text in en) for name, pattern in ACTIONS.items()}
    constructions = {name: sum(match(pattern, text) for text in en)
                     for name, pattern in QUANTITY_PROBES.items()}
    probes = {}
    for label, action, object_rx, target_action, target_object in PROBES:
        action_rows = [row for row, text in zip(rows, en, strict=True) if match(ACTIONS[action], text)]
        object_rows = [row for row, text in zip(rows, en, strict=True) if match(object_rx, text)]
        candidates = [row for row in action_rows if match(object_rx, canonical(row["en"]))]
        aligned_cues = [row for row in candidates
                        if re.search(target_action, canonical(row["hi"])) and
                        re.search(target_object, canonical(row["hi"]))]
        probes[label] = {"action_rows": len(action_rows), "object_rows": len(object_rows),
                         "cooccurrence_ids": [r["id"] for r in candidates],
                         "bilingual_cue_ids": [r["id"] for r in aligned_cues]}
    counts = Counter(row["family"] for row in rows)
    return {"rows": len(rows), "sources": len(sources), "families": len(counts),
            "multireference_sources": sum(len(row["refs"]) > 1 for row in sources),
            "multirow_families": sum(n > 1 for n in counts.values()), "largest_family": max(counts.values()),
            "negation_cues": sum(bool(NEGATIVE.search(row["en"])) for row in rows),
            "questions": sum(row["en"].endswith("?") for row in rows),
            "request_cues": sum(bool(REQUEST.search(text)) for text in en),
            "explicit_or_vague_quantity_cues": sum(bool(QUANTITY.search(text)) for text in en),
            "tracked_name_cues": sum(bool(NAME.search(row["en"])) for row in rows),
            "action_surface_counts": actions, "quantity_surface_counts": constructions, "joint_probes": probes}


def run(args):
    if sha(args.archive) != RESULT_SHA:
        raise ValueError("Pilot 3 archive digest differs")
    rows = load_verified(args.data, ("train", "validation"))
    validation = unique_sources(rows["validation"])
    with ZipFile(args.archive) as archive:
        frozen = json.loads(archive.read("review-sample.json"))
        packet = json.loads(archive.read("reviewer/blind-review.json"))
        predictions = {arm: [json.loads(line) for line in archive.read(
            f"{arm}/validation-1650.jsonl").splitlines()] for arm in ("control", "treatment")}
    if len(packet["items"]) != 30 or any(
            item[key] is not None for item in packet["items"] for key in ("A_adequate", "B_adequate")):
        raise ValueError("Frozen independent review packet changed")
    for arm in predictions.values():
        if [{key: row[key] for key in ("en", "refs", "ids")} for row in arm] != validation:
            raise ValueError("Saved predictions differ from frozen development inputs")
    blind_ids = {row["id"] for row in frozen}
    sample = sample_ids(rows["train"], validation, blind_ids)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    expected = [ident for ids in sample["validation_ids"].values() for ident in ids]
    if [r["id"] for r in annotations["rows"]] != expected:
        raise ValueError("Semantic annotations differ from the seeded analyst sample")
    train_set = {canonical(r["en"]) for r in rows["train"]}
    val_set = {canonical(r["en"]) for r in rows["validation"]}
    train_hi = {canonical(r["hi"]) for r in rows["train"]}
    val_hi = {canonical(r["hi"]) for r in rows["validation"]}
    joined = [dict(r, split=split) for split, entries in rows.items() for r in entries]
    train_words = {word for line in train_set for word in line.split()}
    train_bigrams = {pair for line in train_set for pair in zip(line.split(), line.split()[1:])}
    report = {"dataset_manifest_sha256": sha(args.data / "manifest.json"), "pilot3_archive_sha256": RESULT_SHA,
              "sample": sample, "splits": {name: split_payload(items) for name, items in rows.items()},
              "overlap": {"canonical_english": len(train_set & val_set), "canonical_hindi": len(train_hi & val_hi),
                          "mixed_recomputed_families": sum(len({r["split"] for r in family}) > 1
                                                            for family in families(joined)),
                          "known_token_bigrams_absent_from_train": sum(
                              pair not in train_bigrams for r in rows["validation"]
                              for tokens in [canonical(r["en"]).split()] for pair in zip(tokens, tokens[1:])
                              if all(word in train_words for word in pair))},
              "analyst_error_counts": dict(Counter(label for row in annotations["rows"] for label in row["labels"])),
              "analyst_sample_size": len(expected), "independent_adequacy": "pending",
              "validation_model_metrics": {
                  arm: {**scores(validation, generated), **fact_proxies(validation, generated),
                        **length_stats(validation, generated)} for arm, generated in predictions.items()}}
    write_json(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "data", "annotations", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    run(parser.parse_args())
