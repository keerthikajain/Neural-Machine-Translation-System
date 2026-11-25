"""Read-only diagnostics for the training and validation partitions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from src.data.dataset import collate_fn
from .data import canonical, families, write_json, read_rows
from .evaluate import fact_proxies, load_candidate
from .metrics import PAD, EOS, generate, retrieval, scores, unique_sources
from .train import dataset, load_verified


@torch.inference_mode()
def teacher_metrics(model, tokenizer, rows, limit, batch_size):
    data = dataset(rows, tokenizer, limit)
    nll = tokens = correct = correct_first = correct_eos = early = 0
    for start in range(0, len(data), batch_size):
        batch = collate_fn([data[i] for i in range(start, min(start + batch_size, len(data)))])
        source, target = batch["src_tokens"], batch["tgt_output"]
        logits = model(source, batch["tgt_input"])
        predicted = logits.argmax(dim=-1)
        valid, eos = target.ne(PAD), target.eq(EOS)
        correct += predicted.eq(target).logical_and(valid).sum().item()
        correct_first += predicted[:, 0].eq(target[:, 0]).sum().item()
        correct_eos += predicted.eq(EOS).logical_and(eos).sum().item()
        early += predicted.eq(EOS).logical_and(valid).logical_and(~eos).sum().item()
        nll += torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                                ignore_index=PAD, reduction="sum").item()
        tokens += valid.sum().item()
    return {"nll": nll / tokens, "tokens": tokens, "top1_correct": correct,
            "first_piece_correct": correct_first, "eos_correct": correct_eos,
            "early_eos_at_reference_prefix": early, "pairs": len(rows)}


def cover(texts, tokenizer):
    encoded = [tokenizer.encode(text) for text in texts]
    flat = Counter(token for sentence in encoded for token in sentence)
    byte_ids = {index for index in flat if tokenizer.is_byte(index)}
    lengths = sorted(map(len, encoded))
    return {
        "surface_types": len({word for text in texts for word in canonical(text).split()}),
        "pieces_observed": len(flat), "unknown_pieces": flat[tokenizer.unk_id()],
        "byte_pieces": sum(flat[index] for index in byte_ids),
        "sentences_with_bytes": sum(bool(byte_ids.intersection(sentence)) for sentence in encoded),
        "total_pieces": sum(flat.values()), "median_pieces": lengths[len(lengths) // 2],
        "max_pieces": max(lengths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "evaluation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer, config, step = load_candidate(args.run, args.data, "cpu")
    rows = load_verified(args.data, ("train", "validation"))
    result = {"step": step, "splits": {}}
    for split, records in rows.items():
        result["splits"][split] = {
            "teacher_forced": teacher_metrics(model, tokenizer, records, config["sequence_limit"],
                                              config["batch_size"]),
            **{language: cover([row[language] for row in records], tokenizer)
               for language in ("en", "hi")},
        }
    # Recompute overlap using only permitted partitions.
    combined = [dict(row, split=split) for split, entries in rows.items() for row in entries]
    result["cross_split_families"] = sum(len({r["split"] for r in group}) > 1
                                         for group in families(combined))
    validation = unique_sources(rows["validation"])
    baseline = retrieval(rows["train"], validation)
    result["retrieval_fact_proxies"] = fact_proxies(validation, baseline)
    saved = read_rows(args.evaluation / "validation.jsonl")
    predicted = generate(model, tokenizer, [row["en"] for row in validation],
                         "cpu", config["decode_limit"], config["batch_size"])
    assert all(old["prediction"] == new["prediction"] and old["eos"] == new["eos"]
               for old, new in zip(saved, predicted, strict=True))
    result["replayed_validation"] = scores(validation, predicted)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "diagnostics.json", result)
    write_json(args.output / "retrieval-examples.json",
               [dict(row, **pred) for row, pred in zip(validation, baseline, strict=True)])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
