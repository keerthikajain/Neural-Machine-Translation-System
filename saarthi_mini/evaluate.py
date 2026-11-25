"""Evaluate a frozen candidate. Keep final test predictions out of training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


import sentencepiece as spm
import torch

from .data import canonical, sha, write_json, write_rows
from .metrics import generate, retrieval, scores, unique_sources, verify_tokenizer
from .train import conditioning, dataset, load_verified, make_model


NUMERALS = {
    "zero": "शून्य", "one": "एक", "two": "दो", "three": "तीन", "four": "चार",
    "five": "पाँच पांच", "six": "छह छः छ", "seven": "सात", "eight": "आठ",
    "nine": "नौ", "ten": "दस", "eleven": "ग्यारह", "twelve": "बारह",
}
NAMES = {"tom": "टॉम टाम टोम", "john": "जॉन जान", "mary": "मैरी मेरी",
         "india": "भारत इंडिया हिन्दुस्तान हिंदुस्तान", "delhi": "दिल्ली"}
HI_NUMBERS = {canonical(word): str(i)
              for i, words in enumerate(NUMERALS.values()) for word in words.split()}
EN_NUMBERS = {word: str(i) for i, word in enumerate(NUMERALS)}


def number_set(text, mapping):
    return {token if token.isdecimal() else mapping[token] for token in canonical(text).split()
            if token.isdecimal() or token in mapping}


def fact_proxies(rows, predictions):
    outcomes = Counter()
    for row, item in zip(rows, predictions, strict=True):
        en = set(canonical(row["en"]).split())
        prediction = set(canonical(item["prediction"]).split())
        refs = [set(canonical(ref).split()) for ref in row["refs"]]
        expected = []
        for name, variants in NAMES.items():
            spellings = set(canonical(variants).split())
            if name in en and any(spellings & ref for ref in refs):
                expected.append(spellings)
        if expected:
            outcomes["name_eligible"] += 1
            outcomes["name_pass"] += all(spellings & prediction for spellings in expected)
        numbers = number_set(row["en"], EN_NUMBERS)
        if numbers and any(numbers == number_set(ref, HI_NUMBERS) for ref in row["refs"]):
            outcomes["number_eligible"] += 1
            outcomes["number_pass"] += numbers == number_set(item["prediction"], HI_NUMBERS)
    return {key: outcomes[key] for key in ("name_eligible", "name_pass", "number_eligible", "number_pass")}


def load_candidate(run, data_path, device):
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    if not latest["terminal"] or not latest["best_step"]:
        raise ValueError("Finish the run and select a validation checkpoint before evaluation")
    checkpoint = torch.load(run / f"best-{latest['best_step']}.pt", map_location="cpu",
                            weights_only=True)
    fingerprint = latest["fingerprint"]
    if (checkpoint["fingerprint"] != fingerprint or checkpoint["step"] != latest["best_step"]
            or sha(run / "tokenizer.model") != fingerprint["tokenizer"]
            or sha(run / "config.json") != fingerprint["config"]
            or (data_path is not None and sha(data_path / "manifest.json") != fingerprint["manifest"])):
        raise ValueError("Candidate, tokenizer, configuration or dataset identity mismatch")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    tokenizer = spm.SentencePieceProcessor(model_file=str(run / "tokenizer.model"))
    verify_tokenizer(tokenizer)
    model = make_model(config, tokenizer).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, config, checkpoint["step"]


def review_requests(args, model, tokenizer, config, checkpoint_step, test, predictions, all_rows):
    probes = json.loads(args.challenges.read_text(encoding="utf-8"))["probes"]
    ids = {row["id"]: row for row in all_rows["test"]}
    other_sources = {canonical(row["en"]) for split in ("train", "validation") for row in all_rows[split]}
    for probe in probes:
        if canonical(ids[probe["anchor"]]["en"]) != canonical(probe["variants"][0]):
            raise ValueError(f"Challenge anchor changed for {probe['id']}")
        if "companion" in probe and canonical(ids[probe["companion"]]["en"]) != canonical(probe["variants"][1]):
            raise ValueError(f"Challenge companion changed for {probe['id']}")
        if any(canonical(text) in other_sources for text in probe["variants"]):
            raise ValueError(f"Challenge source crossed splits for {probe['id']}")
    translations = generate(model, tokenizer, [text for probe in probes for text in probe["variants"]],
                            "cpu", config["decode_limit"])
    contrast_rows = []
    for j, probe in enumerate(probes):
        for i, text in enumerate(probe["variants"]):
            contrast_rows.append({"id": f"{probe['id']}-{i+1}", "pair_id": probe["id"],
                                  "kind": probe["kind"], "en": text, "expected_facts": probe["facts"][i],
                                  **translations[j*2+i]})
    by_family = {}
    references = {row["id"]: row["family"] for row in all_rows["test"]}
    review_rows = list(zip(test, predictions, strict=True))
    random.Random(3021).shuffle(review_rows)
    for row, generation in review_rows:
        family = references[row["ids"][0]]
        by_family.setdefault(family, {"id": row["ids"][0], "kind": "adequacy", **row, **generation})
    review = {"checkpoint_step": checkpoint_step, "reviewer": None,
              "items": [{**row, "pass": None, "notes": ""}
                        for row in [*list(by_family.values())[:60], *contrast_rows]]}
    write_json(args.output / "bilingual-review.json", review)



def cpu_gate(validation, limits):
    count = validation["sources"]
    measures = {
        "validation_min_chrf++": validation["chrf++"],
        "validation_min_gain_over_retrieval_chrf++": (
            validation["chrf++"] - validation["retrieval"]["chrf++"]),
        "validation_min_source_delta_nll": validation["source_delta_nll"],
        "validation_max_repeated_word_trigram_rate": validation["repeated_word_trigram"] / count,
        "validation_max_no_eos_rate": validation["no_eos"] / count,
    }
    checks = {}
    for name, actual in measures.items():
        threshold = limits[name]
        passed = actual >= threshold if "_min_" in name else actual <= threshold
        checks[name] = {"actual": actual, "threshold": threshold, "pass": passed}
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}

def evaluate(args):
    torch.set_num_threads(2)
    rows = load_verified(args.data, ("train", "validation"))
    model, tokenizer, config, step = load_candidate(args.run, args.data, "cpu")
    seen = unique_sources(rows["train"])
    random.Random(123).shuffle(seen)
    subjects = {"seen": seen[:200], "validation": unique_sources(rows["validation"])}
    outputs = {}
    report = {"step": step, "splits": {}}
    args.output.mkdir(parents=True, exist_ok=False)
    def measure_split(split, examples):
        predictions = generate(model, tokenizer, [row["en"] for row in examples],
                               "cpu", config["decode_limit"])
        outputs[split] = predictions
        measures = scores(examples, predictions)
        measures["fact_proxies"] = fact_proxies(examples, predictions)
        if split != "seen":
            measures["retrieval"] = scores(examples, retrieval(rows["train"], examples))
            measures.update(conditioning(model, dataset(rows[split], tokenizer, config["sequence_limit"]),
                                         "cpu", config["batch_size"]))
        report["splits"][split] = measures
        write_rows(args.output / f"{split}.jsonl",
                   [dict(row, **pred) for row, pred in zip(examples, predictions, strict=True)])
    for split, examples in subjects.items():
        measure_split(split, examples)
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    report["cpu_candidate_gate"] = cpu_gate(report["splits"]["validation"], gates["cpu_candidate"])
    if args.final_test and report["cpu_candidate_gate"]["pass"]:
        rows.update(load_verified(args.data, ("test",)))
        subjects["test"] = unique_sources(rows["test"])
        measure_split("test", subjects["test"])
        test = report["splits"]["test"]
        report["seen_minus_test_chrf++"] = report["splits"]["seen"]["chrf++"] - test["chrf++"]
        review_requests(args, model, tokenizer, config, step, subjects["test"], outputs["test"], rows)
    elif args.final_test:
        report["final_test"] = "skipped_cpu_gate_failed"
    write_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


def review_scores(path):
    review = json.loads(path.read_text(encoding="utf-8"))
    if not review.get("reviewer") or any(type(item["pass"]) is not bool for item in review["items"]):
        raise ValueError("Enter independent reviewer attribution and a Boolean for every judgment")
    grouped = defaultdict(list)
    for item in review["items"]:
        grouped[(item["kind"], item.get("pair_id", item["id"]))].append(item["pass"])
    totals = defaultdict(lambda: [0, 0])
    for (kind, _), decisions in grouped.items():
        if len(decisions) != (1 if kind == "adequacy" else 2):
            raise ValueError("Incomplete or duplicated adjudication group")
        totals[kind][0] += all(decisions)
        totals[kind][1] += 1
    expected = {"adequacy": 60, "name": 4, "number": 4, "meaning": 4}
    if {kind: pair[1] for kind, pair in totals.items()} != expected:
        raise ValueError("Reviewer must adjudicate all preregistered items")
    print(json.dumps({kind: {"passed": pair[0], "total": pair[1], "rate": pair[0]/pair[1]}
                      for kind, pair in totals.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure")
    for argument in ("data", "run", "output"):
        measure.add_argument(f"--{argument}", type=Path, required=True)
    measure.add_argument("--final-test", action="store_true")
    measure.add_argument("--gates", type=Path, default=Path(__file__).with_name("gates.json"))
    measure.add_argument("--challenges", type=Path, default=Path(__file__).with_name("challenges.json"))
    review = commands.add_parser("score-review")
    review.add_argument("review", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        evaluate(args)
    else:
        review_scores(args.review)


if __name__ == "__main__":
    main()
