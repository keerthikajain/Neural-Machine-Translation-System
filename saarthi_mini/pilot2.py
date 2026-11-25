"""Fork the verified Pilot 1 CPU checkpoint into the bounded Pilot 2 run."""
import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from .data import sha, write_json
from .train import load_verified, save_checkpoint, signature

PARENT_SHA = "0c632830a54601b7534c6299f1c58ed682d64a913d3f860ae6676186d96bd7da"
PARENT_BEST_SHA = "e84b09cd88d890e0e46fd9e9ab291913079cafe41533e84b1423a9e8a3b4186c"


def fork(args):
    parent_file = args.parent / "latest.pt"
    if sha(parent_file) != PARENT_SHA:
        raise ValueError("Pilot 1 checkpoint does not match the preregistered parent")
    parent = torch.load(parent_file, map_location="cpu", weights_only=True)
    best_step = parent["best_step"]
    best_file = args.parent / f"best-{best_step}.pt"
    if sha(best_file) != PARENT_BEST_SHA or parent["step"] != best_step or best_step != 600:
        raise ValueError("Pilot 1 best and terminal checkpoints must both be update 600")
    old_config = args.parent / "config.json"
    tokenizer_path = args.parent / "tokenizer.model"
    load_verified(args.data, ("train", "validation"))
    old = json.loads(old_config.read_text(encoding="utf-8"))
    new = json.loads(args.config.read_text(encoding="utf-8"))
    expected = dict(old, version="mini-pilot-v2-extension", schedule_updates=best_step,
                    max_updates=2000, max_seconds=480)
    if new != expected:
        raise ValueError("Pilot 2 may only extend the update and runtime caps")
    current = signature(SimpleNamespace(data=args.data, config=old_config, device="cpu", threads=2),
                        tokenizer_path)
    original = parent["fingerprint"]
    # Trainer telemetry and schedule extension are the only code changes in this fingerprint.
    current["code"]["saarthi_mini/train.py"] = original["code"]["saarthi_mini/train.py"]
    if current != original:
        raise ValueError("Pilot 1 and Pilot 2 data, tokenizer, runtime or model source differ")
    best = torch.load(best_file, map_location="cpu", weights_only=True)
    if any(not torch.equal(value, best["model"][key]) for key, value in parent["model"].items()):
        raise ValueError("Terminal checkpoint differs from the selected checkpoint")

    args.run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(tokenizer_path, args.run / "tokenizer.model")
    shutil.copyfile(args.config, args.run / "config.json")
    shutil.copyfile(args.parent / "baseline.json", args.run / "baseline.json")
    current = signature(SimpleNamespace(data=args.data, config=args.config, device="cpu", threads=2),
                        tokenizer_path)
    save_checkpoint(args.run / f"best-{best_step}.pt", dict(best, fingerprint=current))
    save_checkpoint(args.run / "latest.pt",
                    dict(parent, fingerprint=current, history=[], terminal=None, seconds=0.0))
    write_json(args.run / "parent.json", {
        "parent_checkpoint_sha256": PARENT_SHA, "parent_best_sha256": PARENT_BEST_SHA,
        "initial_update": best_step, "initial_validation_chrf++": parent["best"],
        "pilot2_config_sha256": sha(args.config), "tokenizer_sha256": sha(tokenizer_path),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "data", "run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot2.json"))
    fork(parser.parse_args())


if __name__ == "__main__":
    main()
