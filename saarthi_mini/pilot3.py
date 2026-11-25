"""Run paired CPU-only learning-rate continuations from the pinned update 600."""
import argparse
import json
from pathlib import Path
import random
import resource
import shutil
import signal
import time
from types import SimpleNamespace

import torch

from .data import sha, write_json, write_rows
from .decode_diagnostics import length_stats
from .evaluate import fact_proxies
from .metrics import unique_sources
from .pilot2 import PARENT_BEST_SHA, PARENT_SHA
from .train import load_verified, save_checkpoint, signature, train

START = 600
CAP = 1650
SECONDS = 300
RSS_KIB = 768 * 1024
RATES = {"control": 0.0001, "treatment": 0.0003}
BASE_NLL = 3.003686478586191


def sample_review(validation):
    rng = random.Random(5032026)
    groups = {"number": [], "name": [], "other": []}
    for row in validation:
        counts = fact_proxies([row], [{"prediction": ""}])
        group = "number" if counts["number_eligible"] else (
            "name" if counts["name_eligible"] else "other")
        groups[group].append(row)
    if len(groups["number"]) != 11:
        raise ValueError("Unexpected frozen numeric stratum")
    selected = [(group, row) for group, size in (("number", 11), ("name", 9), ("other", 10))
                for row in rng.sample(groups[group], size)]
    rng.shuffle(selected)
    return [{"id": row["ids"][0], "stratum": group, "en": row["en"], "refs": row["refs"]}
            for group, row in selected]


def prepare(args):
    old = args.parent / "config.json"
    tokenizer = args.parent / "tokenizer.model"
    files = {"latest.pt": PARENT_SHA, "best-600.pt": PARENT_BEST_SHA}
    for name, fingerprint in files.items():
        if sha(args.parent / name) != fingerprint:
            raise ValueError(f"Unexpected parent digest for {name}")
    saved, best = [torch.load(args.parent / name, map_location="cpu", weights_only=True)
                   for name in files]
    if saved["step"] != START or best["step"] != START or saved["best_step"] != START:
        raise ValueError("Parent checkpoints are not the selected update-600 state")
    if saved["history"][-1]["nll"] != BASE_NLL:
        raise ValueError("Unexpected update-600 NLL")
    if best["fingerprint"] != saved["fingerprint"] or any(
            not torch.equal(saved["model"][name], weight) for name, weight in best["model"].items()):
        raise ValueError("Selected and resumable parent weights differ")
    state, groups = saved["optimizer"]["state"], saved["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0]["lr"] != RATES["control"] or not state or any(
            int(buffer["step"].item()) != START for buffer in state.values()):
        raise ValueError("Unexpected parent AdamW state")
    identity = SimpleNamespace(data=args.data, config=old, device="cpu", threads=2)
    implementation = signature(identity, tokenizer)
    old_fingerprint = saved["fingerprint"]
    implementation["code"]["saarthi_mini/train.py"] = old_fingerprint["code"]["saarthi_mini/train.py"]
    if implementation != old_fingerprint:
        raise ValueError("Mismatch in data, model, tokenizer or runtime provenance")
    rows = load_verified(args.data, ("train", "validation"))
    validation = unique_sources(rows["validation"])
    review = sample_review(validation)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "review-sample.json", review)
    write_json(args.output / "provenance.json", {
        "parent": files, "parent_fingerprint": old_fingerprint, "initial_optimizer_lr": groups[0]["lr"],
        "optimizer_parameter_states": len(state), "optimizer_step": START, "review_seed": 5032026,
        "review_sample_sha256": sha(args.output / "review-sample.json"),
        "baseline_nll": BASE_NLL, "baseline_chrf++": saved["best"],
    })
    original = json.loads(old.read_text(encoding="utf-8"))
    for arm, rate in RATES.items():
        run = args.output / arm
        run.mkdir()
        config = run / "config.json"
        write_json(config, dict(original, version="mini-pilot-v3-matched-lr", schedule_updates=START,
                                max_updates=CAP, max_seconds=SECONDS, continuation_lr=rate))
        shutil.copyfile(tokenizer, run / "tokenizer.model")
        current = signature(SimpleNamespace(data=args.data, config=config, device="cpu", threads=2),
                            run / "tokenizer.model")
        save_checkpoint(run / "initial.pt", dict(saved, fingerprint=current, history=[], seconds=0.0,
                                                   terminal=None, stale=0, train_window=(0.0, 0)))
        save_checkpoint(run / "best-600.pt", dict(best, fingerprint=current))
        shutil.copyfile(run / "initial.pt", run / "latest.pt")


def divergence_streak(previous, current_nll):
    return previous + 1 if current_nll > BASE_NLL + 0.3 else 0


def resource_peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_arm(args):
    start = time.monotonic()
    directory = args.output / args.arm
    config = directory / "config.json"
    arm_config = json.loads(config.read_text(encoding="utf-8"))
    if arm_config["continuation_lr"] != RATES[args.arm]:
        raise ValueError("Unexpected learning rate for arm")
    if (directory / "runtime.json").exists():
        raise ValueError("Arm has already run")
    validation = unique_sources(load_verified(args.data, ("validation",))["validation"])
    diverging = 0

    def resource_stop():
        return "rss_budget" if resource_peak() >= RSS_KIB else None

    def observe(step, predictions, report):
        nonlocal diverging
        report["lr"] = arm_config["continuation_lr"]
        report.update(fact_proxies(validation, predictions))
        report.update(length_stats(validation, predictions))
        write_rows(directory / f"validation-{step}.jsonl",
                   [dict(row, **prediction) for row, prediction in zip(validation, predictions, strict=True)])
        diverging = divergence_streak(diverging, report["nll"])
        if diverging == 2:
            return "validation_divergence"
        return resource_stop()

    def deadline(signum, frame):
        raise TimeoutError("300-second elapsed budget")

    status = "complete"
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, SECONDS - (time.monotonic() - start)))
    try:
        train(SimpleNamespace(config=config, data=args.data, run=directory, device=torch.device("cpu"),
                              threads=2, resume=True, pause_after=None),
              validation_observer=observe, resource_stop=resource_stop)
    except TimeoutError:
        status = "hard_time_budget"
    except (FloatingPointError, RuntimeError) as error:
        status = "numerical_stop" if "non-finite" in str(error) or "Nonfinite" in str(error) else "error"
        if status == "error":
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        write_json(directory / "runtime.json", {"status": status,
                   "elapsed_seconds": time.monotonic() - start, "peak_rss_kib": resource_peak()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", *RATES))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args)
    else:
        args.arm = args.action
        run_arm(args)


if __name__ == "__main__":
    main()
