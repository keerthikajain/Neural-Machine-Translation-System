"""Compare completed Pilot 3 validation checkpoints and prepare blind review."""
import argparse
import json
from pathlib import Path
import random

from .data import read_rows, sha, write_json
from .evaluate import cpu_gate
from .pilot3 import RATES, START


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_step(rows, baseline, delta):
    selected = None
    for step, record in sorted(rows.items()):
        if record["chrf++"] >= baseline + delta:
            selected = step
            baseline = record["chrf++"]
    return selected


def blind_packet(sample, predictions):
    rng = random.Random(5032027)
    packet, key = [], {}
    for index, item in enumerate(sample, 1):
        arm_order = rng.sample(list(RATES), len(RATES))
        row = {"item": index, "english": item["en"], "references": item["refs"]}
        for label, arm in zip(("A", "B"), arm_order, strict=True):
            row[label] = predictions[arm][item["id"]]["prediction"]
            row[f"{label}_adequate"] = None
        row["notes"] = ""
        key[str(index)] = dict(zip(("A", "B"), arm_order, strict=True))
        packet.append(row)
    return packet, key


def compare(path):
    provenance = load_json(path / "provenance.json")
    sample_path = path / "review-sample.json"
    if sha(sample_path) != provenance["review_sample_sha256"]:
        raise ValueError("Frozen bilingual review sample changed")
    histories, report = {}, {"arms": {}, "review_status": "unverified"}
    report["provenance"] = provenance
    for arm, rate in RATES.items():
        directory = path / arm
        config = load_json(directory / "config.json")
        if config["continuation_lr"] != rate:
            raise ValueError("Unexpected learning rate")
        history = load_json(directory / "history.json")
        runtime = load_json(directory / "runtime.json")
        records = {row["step"]: row for row in history["epochs"]}
        histories[arm] = records
        report["arms"][arm] = {
            "config": config, "runtime": runtime, "history": list(records.values()),
            "terminal_reason": history["terminal"] if runtime["status"] == "complete" else runtime["status"],
            "final_checkpoint_sha256": sha(directory / "latest.pt"),
            "best_checkpoint_sha256": sha(directory / f"best-{history['best_step']}.pt"),
            "initial_checkpoint_sha256": sha(directory / "initial.pt"),
            "last_validated_step": max(records, default=START), "accepted_best_step": history["best_step"],
            "best_raw_step": max(records, key=lambda step: records[step]["chrf++"], default=START),
        }
    control, treatment = histories.values()
    shared = sorted(control.keys() & treatment.keys())
    report["shared_steps"] = shared
    report["treatment_selected_step"] = selected = select_step(
        {step: treatment[step] for step in shared}, provenance["baseline_chrf++"],
        report["arms"]["treatment"]["config"]["min_delta_chrf"])
    if selected is not None:
        control_row, treatment_row = control[selected], treatment[selected]
        chrf_gain = treatment_row["chrf++"] - control_row["chrf++"]
        report["acceptance_checks"] = {
            "matched_chrf_gain": chrf_gain, "matched_chrf_gate_pass": chrf_gain >= 3,
            "number_gain": treatment_row["number_pass"] - control_row["number_pass"],
            "numeric_gate_pass": treatment_row["number_pass"] >= 3 and (
                treatment_row["number_pass"] > control_row["number_pass"]),
            "factual_adequacy_gate_pass": None,
        }
        limits = load_json(Path(__file__).with_name("gates.json"))["cpu_candidate"]
        retrieval = load_json(path / "control-retrieval.json")
        report["cpu_gate"] = cpu_gate(dict(treatment_row, retrieval=retrieval), limits)
        predictions = {arm: {row["ids"][0]: row for row in read_rows(path / arm / f"validation-{selected}.jsonl")}
                       for arm in RATES}
        packet, key = blind_packet(load_json(sample_path), predictions)
        write_json(path / "blind-review.json", {
            "rubric": "Mark A and B independently true or false for preservation of the main action or state, "
                      "participants, roles, material objects and details, polarity, names and numbers, "
                      "without inventing material facts. Accept faithful paraphrase. If a reference conflicts "
                      "with English, judge the meaning of the English and note the conflict.",
            "minimum_net_treatment_wins": 6, "items": packet,
        })
        write_json(path / "blind-review-key.json", key)
    report["decision"] = "NO-GO pending all predeclared gates including independent bilingual review"
    write_json(path / "comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    compare(parser.parse_args().output)
