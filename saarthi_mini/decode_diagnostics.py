"""Run fixed-weight decoding comparisons on the development split only."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import canonical, families, read_rows, sha, write_json, write_rows
from .diagnose import teacher_metrics
from .evaluate import fact_proxies, load_candidate
from .metrics import BOS, EOS, PAD, UNK, generate, retrieval, scores, unique_sources
from .train import conditioning, dataset, learning_rate, load_verified


def block_trigram(logits, prefixes):
    """Suppress tokens that would repeat a generated token trigram."""
    for row, prefix in enumerate(prefixes.tolist()):
        prefix = prefix[1:]  # BOS is outside the translated sentence.
        if len(prefix) < 3:
            continue
        blocked = {c for a, b, c in zip(prefix, prefix[1:], prefix[2:])
                   if [a, b] == prefix[-2:]}
        if blocked:
            logits[row, list(blocked)] = -torch.inf


def lengths_and_eos(tokens):
    eos = tokens.eq(EOS)
    ended = eos.any(-1)
    length = (eos.long().argmax(-1) + 1).masked_fill(~ended, tokens.size(-1))
    return length, ended


@torch.inference_mode()
def candidates(model, tokenizer, texts, limit, *, width=1, batch_size=32,
               minimum=0, no_repeat=False):
    """Return beam candidates with raw cumulative log probability.

    Finished paths keep their score and acquire padding only. Length is scored
    separately from beam expansion so both rerankers see identical candidates.
    """
    model.eval()
    if width < 1 or minimum < 0 or not minimum < limit:
        raise ValueError("Invalid decoding bounds")
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        items = [torch.tensor([BOS, *tokenizer.encode(text)]) for text in batch]
        position_limit = model.encoder.embedding.positional_encoding.pe.size(1)
        if max(map(len, items)) > position_limit or limit > position_limit:
            raise ValueError("Decode limit exceeds model positions")
        source = pad_sequence(items, batch_first=True, padding_value=PAD)
        memory, source_mask = model.encode(source)
        memory = memory.repeat_interleave(width, dim=0)
        source_mask = source_mask.repeat_interleave(width, dim=0)
        size = len(items)
        prefixes = source.new_full((size, width, 1), BOS)
        accumulated = memory.new_full((size, width), -torch.inf)
        accumulated[:, 0] = 0
        finished = source.new_zeros((size, width), dtype=torch.bool)
        for step in range(limit):
            logits = model.decode_step(prefixes.flatten(0, 1), memory, source_mask)[:, -1, :]
            logits[:, [PAD, UNK, BOS]] = -torch.inf
            probabilities = logits.log_softmax(dim=-1)
            # Apply constraints after normalization so remaining path scores
            # reflect the model's original preferences.
            if step < minimum:
                probabilities[:, EOS] = -torch.inf
            if no_repeat:
                block_trigram(probabilities, prefixes.flatten(0, 1))
            done = finished.flatten()
            probabilities[done] = -torch.inf
            probabilities[done, PAD] = 0
            vocabulary = probabilities.size(-1)
            proposed = accumulated.unsqueeze(-1) + probabilities.reshape(size, width, vocabulary)
            accumulated, best = proposed.flatten(1).topk(width, dim=-1)
            parents, new = best.div(vocabulary, rounding_mode="floor"), best.remainder(vocabulary)
            prefixes = prefixes.gather(1, parents.unsqueeze(-1).expand(-1, -1, prefixes.size(-1)))
            prefixes = torch.cat((prefixes, new.unsqueeze(-1)), dim=-1)
            finished = finished.gather(1, parents) | new.eq(EOS)
            if (finished | ~accumulated.isfinite()).all():
                break
        yield prefixes[:, :, 1:], accumulated


def select(tokenizer, batches, alpha=0):
    results = []
    for tokens, likelihood in batches:
        lengths, ended = lengths_and_eos(tokens)
        ranks = likelihood / lengths.float().pow(alpha)
        batch_indices = torch.arange(tokens.shape[0])
        chosen = ranks.argmax(-1)
        sequence = tokens[batch_indices, chosen].tolist()
        sizes = lengths[batch_indices, chosen].tolist()
        eos = ended[batch_indices, chosen].tolist()
        for ids, length, stop in zip(sequence, sizes, eos, strict=True):
            results.append({"prediction": tokenizer.decode(ids[:length - int(stop)]), "eos": stop})
    return results


def length_stats(rows, generations):
    predicted = [len(canonical(g["prediction"]).split()) for g in generations]
    reference = [mean(len(canonical(ref).split()) for ref in row["refs"]) for row in rows]
    return {
        "mean_words": mean(predicted), "median_words": median(predicted),
        "min_words": min(predicted), "max_words": max(predicted),
        "reference_mean_words": mean(reference),
        "corpus_word_length_ratio": sum(predicted) / sum(reference),
        "eos_below_half_shortest_reference": sum(
            g.get("eos") is True and 2 * length < min(len(canonical(ref).split()) for ref in row["refs"])
            for row, g, length in zip(rows, generations, predicted, strict=True)),
    }


def coverage(train, validation, tokenizer):
    result = {}
    for language in ("en", "hi"):
        def words(rows):
            return [canonical(row[language]).split() for row in rows]

        train_words, val_words = words(train), words(validation)
        counts = Counter(word for sequence in train_words for word in sequence)
        val_count = Counter(word for sequence in val_words for word in sequence)
        train_bigrams = {pair for sequence in train_words for pair in zip(sequence, sequence[1:])}
        all_bigrams = [pair for sequence in val_words for pair in zip(sequence, sequence[1:])]
        encoded = [tokenizer.encode(row[language]) for row in validation]
        known_bigrams = [pair for pair in all_bigrams if all(word in counts for word in pair)]
        result[language] = {
            "train_types": len(counts), "train_singleton_types": sum(n == 1 for n in counts.values()),
            "validation_types": len(val_count), "validation_words": sum(val_count.values()),
            "validation_unseen_types": len(val_count.keys() - counts.keys()),
            "validation_unseen_word_tokens": sum(n for word, n in val_count.items() if word not in counts),
            "validation_singleton_word_tokens": sum(n for word, n in val_count.items() if counts[word] == 1),
            "validation_sentences_with_unseen_words": sum(any(w not in counts for w in ws) for ws in val_words),
            "validation_bigrams": len(all_bigrams),
            "validation_unseen_bigrams": sum(pair not in train_bigrams for pair in all_bigrams),
            "validation_bigrams_with_seen_words": len(known_bigrams),
            "validation_unseen_bigrams_with_seen_words": sum(pair not in train_bigrams for pair in known_bigrams),
            "max_validation_pieces": max(map(len, encoded)),
            "unknown_pieces": sum(tok == UNK for sentence in encoded for tok in sentence),
            "byte_pieces": sum(tokenizer.is_byte(tok) for sentence in encoded for tok in sentence),
        }
    return result


@torch.inference_mode()
def eos_probe(model, tokenizer, rows, limit, batch_size):
    # Teacher-forced behavior is invariant across decoding policies.
    return teacher_metrics(model, tokenizer, rows, limit, batch_size)


def run(args):
    torch.set_num_threads(2)
    rows = load_verified(args.data, ("train", "validation"))
    model, tokenizer, config, step = load_candidate(args.run, args.data, "cpu")
    validation = unique_sources(rows["validation"])
    texts = [row["en"] for row in validation]
    offset = len(texts) // 2
    swapped = texts[offset:] + texts[:offset]
    limit, batch = config["decode_limit"], config["batch_size"]
    conditions = conditioning(model, dataset(rows["validation"], tokenizer, config["sequence_limit"]),
                              "cpu", batch)
    latest = torch.load(args.run / "latest.pt", weights_only=True, map_location="cpu")
    report = {
        "checkpoint_step": step, "checkpoint_sha256": sha(args.run / f"best-{step}.pt"),
        "dataset_manifest_sha256": sha(args.data / "manifest.json"), "threads": 2, "batch_size": batch,
        "torch": str(torch.__version__), "source_swap_available_for_all_model_arms": True,
        "conditioning": conditions,
        "teacher_forced": eos_probe(model, tokenizer, rows["validation"], config["sequence_limit"], batch),
        "coverage": coverage(rows["train"], rows["validation"], tokenizer),
        "development_cross_split_families": sum(
            len({row["split"] for row in group}) > 1 for group in families(
                [dict(row, split=split) for split, entries in rows.items() for row in entries])),
        "optimizer": {
            "lr_from_latest_optimizer": latest["optimizer"]["param_groups"][0]["lr"],
            "selected_schedule_lr": learning_rate(step, config),
            "terminal_step": latest["step"], "gradient_clip": 1.0,
            "validation_history": latest["history"],
        },
        "comparisons": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    def measure(name, predictions, altered, seconds, parameters):
        entry = {**scores(validation, predictions), **fact_proxies(validation, predictions),
                 **length_stats(validation, predictions), "settings": parameters,
                 "cpu_seconds": seconds, "ms_per_source": 1000 * seconds / len(validation) if seconds else None}
        if altered is not None:
            entry["rotated_source_chrf++"] = scores(validation, altered)["chrf++"]
            entry["unchanged_after_source_rotation"] = sum(
                canonical(a["prediction"]) == canonical(b["prediction"])
                for a, b in zip(predictions, altered, strict=True))
        report["comparisons"][name] = entry
        write_rows(args.output / f"{name}.jsonl",
                   [dict(row, **pred) for row, pred in zip(validation, predictions, strict=True)])

    generate(model, tokenizer, texts[:batch], "cpu", limit, batch)  # Warm-up outside measurement.
    start = time.perf_counter()
    greedy = generate(model, tokenizer, texts, "cpu", limit, batch)
    elapsed = time.perf_counter() - start
    original = read_rows(args.original)
    if any(old["en"] != source or old["prediction"] != new["prediction"] or old["eos"] != new["eos"]
           for old, source, new in zip(original, texts, greedy, strict=True)):
        raise ValueError("Greedy output differs from the verified Pilot 2 record")
    measure("greedy", greedy, generate(model, tokenizer, swapped, "cpu", limit, batch),
            elapsed, {"width": 1, "maximum_tokens": limit})
    for name, params in (("greedy_token_trigram_block", {"no_repeat": True}),
                         ("greedy_min_5", {"minimum": 5})):
        start = time.perf_counter()
        predicted = select(tokenizer, candidates(model, tokenizer, texts, limit, batch_size=batch, **params))
        elapsed = time.perf_counter() - start
        changed = select(tokenizer, candidates(model, tokenizer, swapped, limit, batch_size=batch, **params))
        measure(name, predicted, changed, elapsed, {"width": 1, "maximum_tokens": limit, **params})
    for name, cap in (("greedy_cap_24", 24), ("greedy_cap_56", 56)):
        start = time.perf_counter()
        predicted = generate(model, tokenizer, texts, "cpu", cap, batch)
        elapsed = time.perf_counter() - start
        measure(name, predicted, generate(model, tokenizer, swapped, "cpu", cap, batch), elapsed,
                {"width": 1, "maximum_tokens": cap})
    start = time.perf_counter()
    pool = list(candidates(model, tokenizer, texts, limit, width=3, batch_size=batch))
    search_time = time.perf_counter() - start
    changed = list(candidates(model, tokenizer, swapped, limit, width=3, batch_size=batch))
    for name, alpha in (("beam_3_raw", 0), ("beam_3_length_rerank", 0.7)):
        start = time.perf_counter()
        predicted = select(tokenizer, pool, alpha)
        elapsed = search_time + time.perf_counter() - start
        measure(name, predicted, select(tokenizer, changed, alpha), elapsed,
                {"width": 3, "maximum_tokens": limit, "length_alpha": alpha})
    baseline = retrieval(rows["train"], validation)
    measure("retrieval", baseline, None, None, {"train_only": True})
    write_json(args.output / "metrics.json", report)
    for name, result in report["comparisons"].items():
        print(name, {key: result[key] for key in ("bleu", "chrf++", "exact", "repeated_word_trigram",
                     "no_eos", "name_pass", "number_pass", "corpus_word_length_ratio", "ms_per_source")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output", "original"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
