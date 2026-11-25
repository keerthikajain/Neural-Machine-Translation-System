"""Run or resume a bounded, scratch-initialized Mini training experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import sentencepiece as spm
import torch
from torch.nn import functional as F

from src.data.dataset import NMTDataset, collate_fn
from src.model.transformer import create_transformer_model

from .data import read_rows, sha, write_json
from .metrics import PAD, generate, retrieval, scores, unique_sources, verify_tokenizer


def load_verified(directory, splits):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    for split in splits:
        path = directory / f"{split}.jsonl"
        if sha(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Dataset integrity check failed for {split}")
        rows[split] = read_rows(path)
    return rows


def dataset(rows, tokenizer, limit):
    encoded = [tokenizer.encode(row[lang]) for row in rows for lang in ("en", "hi")]
    if max(map(len, encoded)) > limit - 2:
        raise ValueError("Dataset exceeds configured token limit. No text was truncated.")
    return NMTDataset(english_sentences=[row["en"] for row in rows],
                      hindi_sentences=[row["hi"] for row in rows],
                      tokenizer=tokenizer, max_length=limit)


def fit_tokenizer(rows, output, vocab_size):
    rows = [row[lang] for row in rows for lang in ("en", "hi")]
    with (output / "tokenizer.model").open("xb") as model_file:
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(rows), model_writer=model_file,
            vocab_size=vocab_size, model_type="unigram", byte_fallback=True,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=False,
            num_threads=1, shuffle_input_sentence=False, character_coverage=1.0,
            minloglevel=2,
        )


def make_model(config, tokenizer):
    return create_transformer_model(vocab_size=tokenizer.vocab_size(),
                                    max_seq_length=config["sequence_limit"], **config["model"])


def losses(model, batch, device, swapped_source=None):
    source = batch["src_tokens"].to(device) if swapped_source is None else swapped_source.to(device)
    logits = model(source, batch["tgt_input"].to(device))
    targets = batch["tgt_output"].to(device)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                           ignore_index=PAD, reduction="sum"), targets.ne(PAD).sum()


@torch.inference_mode()
def conditioning(model, data, device, batch_size):
    model.eval()
    # Move the entire source by half a split, avoiding adjacent length-sorted pairs.
    offsets = [len(data) // 2, len(data) // 3]
    totals, tokens = [0.0] * (len(offsets) + 1), 0
    for start in range(0, len(data), batch_size):
        indices = range(start, min(start + batch_size, len(data)))
        batch = collate_fn([data[i] for i in indices])
        total, count = losses(model, batch, device)
        totals[0] += total.item()
        tokens += count.item()
        for j, offset in enumerate(offsets, 1):
            swapped = collate_fn([data[(i + offset) % len(data)] for i in indices])["src_tokens"]
            total, _ = losses(model, batch, device, swapped_source=swapped)
            totals[j] += total.item()
    proper = totals[0] / tokens
    swapped = sum(totals[1:]) / (len(offsets) * tokens)
    return {"nll": proper, "shuffled_nll": swapped, "source_delta_nll": swapped - proper}


def learning_rate(update, config):
    if "continuation_lr" in config and update > config["schedule_updates"]:
        return config["continuation_lr"]
    warmup = config["warmup_updates"]
    total = config.get("schedule_updates", config["max_updates"])
    update = min(update, total)
    factor = update / warmup if update <= warmup else (
        0.1 + 0.9 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup))) / 2)
    return factor * config["peak_lr"]


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def signature(args, tokenizer_path):
    code = Path(__file__).parent
    repo = code.parent
    implementation = [code / f"{name}.py" for name in ("train", "data", "metrics")]
    implementation.extend(sorted((repo / "src" / "model").glob("*.py")))
    implementation.append(repo / "src" / "data" / "dataset.py")
    pilot3 = json.loads(args.config.read_text(encoding="utf-8")).get("continuation_lr")
    return {
        **({"pilot3_runner": sha(code / "pilot3.py")} if pilot3 is not None else {}),
        "manifest": sha(args.data / "manifest.json"), "tokenizer": sha(tokenizer_path),
        "config": sha(args.config), "torch": str(torch.__version__), "sentencepiece": spm.__version__,
        "device": str(args.device), "threads": args.threads,
        "code": {str(path.relative_to(repo)): sha(path) for path in implementation},
    }


def train(args, *, validation_observer=None, resource_stop=None):
    start = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_verified(args.data, ("train", "validation"))
    seed = config["seed"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    if args.device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not args.allow_cuda:
            raise ValueError("GPU execution requires --allow-cuda")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    if not args.resume:
        args.run.mkdir(parents=True, exist_ok=False)
        fit_tokenizer(rows["train"], args.run, config["vocab_size"])
        (args.run / "config.json").write_bytes(args.config.read_bytes())
    tokenizer_path = args.run / "tokenizer.model"
    fingerprint = signature(args, tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    verify_tokenizer(tokenizer)
    train_data, val_data = [dataset(rows[split], tokenizer, config["sequence_limit"])
                            for split in ("train", "validation")]
    validation = unique_sources(rows["validation"])
    model = make_model(config, tokenizer).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["peak_lr"],
                                  betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    step, best, best_step, stale, history, consumed = 0, -1.0, 0, 0, [], 0.0
    if args.resume:
        checkpoint = torch.load(args.run / "latest.pt", map_location="cpu", weights_only=True)
        legacy = {**fingerprint, "code": {**fingerprint["code"],
                  "saarthi_mini/train.py": "a9511d71fa311cf3f6e22f8e17e21e9fc67be2e31167f592343f9081b333240d"}}
        if checkpoint["fingerprint"] != fingerprint and not (
                "schedule_updates" not in config and checkpoint["fingerprint"] == legacy):
            raise ValueError("Checkpoint data, tokenizer, code or configuration differs")
        fingerprint = checkpoint["fingerprint"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        if args.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, best, best_step, stale, history, consumed = (
            checkpoint[k] for k in ("step", "best", "best_step", "stale", "history", "seconds"))
        if checkpoint["terminal"]:
            print(f"Run already ended at step {step}: {checkpoint['terminal']}", flush=True)
            return

    train_loss_sum, train_tokens = checkpoint.get("train_window", (0.0, 0)) if args.resume else (0.0, 0)

    def elapsed():
        return consumed + time.monotonic() - start

    def save(terminal=None, improved=False):
        # Save the immutable best step first, then atomically commit the resume pointer.
        state = {
            "fingerprint": fingerprint, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "best": best, "best_step": best_step, "stale": stale,
            "history": history, "seconds": elapsed(), "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if args.device.type == "cuda" else [],
            "terminal": terminal, "train_window": (train_loss_sum, train_tokens),
        }
        if improved:
            save_checkpoint(args.run / f"best-{step}.pt", {"model": state["model"],
                                                           "fingerprint": fingerprint, "step": step})
        save_checkpoint(args.run / "latest.pt", state)
        write_json(args.run / "history.json", {"epochs": history, "best_step": best_step,
                                              "terminal": terminal, "train_window": (train_loss_sum, train_tokens), "seconds": round(elapsed(), 2)})

    if not args.resume:
        baseline = scores(validation, retrieval(rows["train"], validation))
        write_json(args.run / "baseline.json", baseline)
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                          "vocabulary": tokenizer.vocab_size(), "retrieval_validation": baseline}), flush=True)
        save()

    batch_size = config["batch_size"]
    per_epoch = math.ceil(len(train_data) / batch_size)
    epoch = -1
    stop_at = min(args.pause_after or config["max_updates"], config["max_updates"])
    while step < stop_at:
        if resource_stop and (reason := resource_stop()):
            save(reason)
            break
        if elapsed() >= config["max_seconds"]:
            save("time_budget")
            break
        next_epoch, batch_index = divmod(step, per_epoch)
        if next_epoch != epoch:
            epoch = next_epoch
            order = list(range(len(train_data)))
            random.Random(seed + epoch).shuffle(order)
        start_idx = batch_index * batch_size
        batch = collate_fn([train_data[i] for i in order[start_idx:start_idx + batch_size]])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, tokens = losses(model, batch, args.device)
        loss = total / tokens
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {step + 1}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.param_groups[0]["lr"] = learning_rate(step + 1, config)
        optimizer.step()
        train_loss_sum += total.detach().item()
        train_tokens += tokens.item()
        step += 1
        if step % config["evaluate_every"] != 0 and step != config["max_updates"]:
            continue
        predictions = generate(model, tokenizer, [row["en"] for row in validation],
                               args.device, config["decode_limit"], batch_size)
        result = scores(validation, predictions)
        result.update(conditioning(model, val_data, args.device, batch_size))
        result.update(step=step, train_batch_nll=loss.item(), grad_norm=grad.item(),
                      train_window_nll=train_loss_sum / train_tokens, train_window_tokens=train_tokens,
                      seconds=round(elapsed(), 2))
        train_loss_sum, train_tokens = 0.0, 0
        history.append(result)
        improved = result["chrf++"] >= best + config["min_delta_chrf"]
        if improved:
            best, best_step, stale = result["chrf++"], step, 0
        else:
            stale += 1
        terminal = None
        if step >= config["max_updates"]:
            terminal = "update_budget"
        elif stale >= config["patience"]:
            terminal = "early_stop"
        if validation_observer:
            terminal = validation_observer(step, predictions, result) or terminal
        save(terminal, improved)
        print(json.dumps(result), flush=True)
        if terminal:
            break
    else:
        if step == stop_at and step < config["max_updates"]:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", type=int)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    if args.threads < 1 or (args.pause_after is not None and args.pause_after < 1):
        parser.error("Thread and pause counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
