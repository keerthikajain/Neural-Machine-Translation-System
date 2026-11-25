"""Generation, surface metrics and train-only retrieval for Saarthi Mini."""
from collections import Counter, defaultdict
import math

from sacrebleu.metrics import BLEU, CHRF
from sentencepiece import SentencePieceProcessor
import torch
from torch.nn.utils.rnn import pad_sequence

from .data import canonical

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def verify_tokenizer(tokenizer: SentencePieceProcessor):
    ids = tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()
    if ids != (PAD, UNK, BOS, EOS):
        raise ValueError(f"Unexpected tokenizer IDs: {ids}")


@torch.inference_mode()
def generate(model, tokenizer, texts, device, limit, batch_size=32):
    model.eval()
    results = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        items = [torch.tensor([BOS, *tokenizer.encode(text)], device=device) for text in batch]
        if max(map(len, items)) > model.encoder.embedding.positional_encoding.pe.size(1):
            raise ValueError("Source exceeds positional limit")
        source = pad_sequence(items, batch_first=True, padding_value=PAD)
        memory, source_mask = model.encode(source)
        target = source.new_full((len(batch), 1), BOS)
        finished = source.new_zeros(len(batch), dtype=torch.bool)
        for _ in range(limit):
            logits = model.decode_step(target, memory, source_mask)[:, -1, :]
            logits[:, [PAD, UNK, BOS]] = -torch.inf
            next_ids = logits.argmax(-1).masked_fill(finished, PAD)
            finished |= next_ids.eq(EOS)
            target = torch.cat((target, next_ids[:, None]), dim=1)
            if finished.all():
                break
        for ids, eos in zip(target[:, 1:].tolist(), finished.tolist(), strict=True):
            tokens = ids[:ids.index(EOS)] if eos else ids
            results.append({"prediction": tokenizer.decode(tokens), "eos": eos})
    return results


def unique_sources(rows):
    grouped = {}
    for row in rows:
        key = canonical(row["en"])
        record = grouped.setdefault(key, {"en": row["en"], "refs": [], "ids": []})
        record["refs"].append(row["hi"])
        record["ids"].append(row["id"])
    return list(grouped.values())


def scores(rows, generations):
    predictions = [item["prediction"] for item in generations]
    references = [row["refs"] for row in rows]
    # SacreBLEU accepts missing alternatives as None and excludes them per sentence.
    reference_streams = [
        [group[i] if i < len(group) else None for group in references]
        for i in range(max(map(len, references)))
    ]
    bleu, chrf = BLEU(tokenize="13a"), CHRF(word_order=2)
    repeat = 0
    for text in predictions:
        words = canonical(text).split()
        trigrams = list(zip(words, words[1:], words[2:]))
        repeat += len(set(trigrams)) < len(trigrams)
    return {
        "sources": len(rows),
        "bleu": bleu.corpus_score(predictions, reference_streams).score,
        "chrf++": chrf.corpus_score(predictions, reference_streams).score,
        "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
        "exact": sum(canonical(pred) in {canonical(ref) for ref in refs}
                     for pred, refs in zip(predictions, references, strict=True)),
        "repeated_word_trigram": repeat,
        "empty": sum(not text.strip() for text in predictions),
        "no_eos": sum(item.get("eos") is False for item in generations),
    }


def grams(text):
    padded = " " + canonical(text) + " "
    return Counter(padded[i:i+3] for i in range(len(padded) - 2))


def retrieval(train_rows, eval_rows):
    index = defaultdict(list)
    for idx, row in enumerate(train_rows):
        features = grams(row["en"])
        norm = math.sqrt(sum(n*n for n in features.values()))
        for key, value in features.items():
            index[key].append((idx, value / norm))
    predictions = []
    for row in eval_rows:
        similarity = defaultdict(float)
        for key, value in grams(row["en"]).items():
            for idx, normalized in index[key]:
                similarity[idx] += value * normalized
        chosen = max(similarity, key=lambda i: (similarity[i], -i), default=0)
        predictions.append({"prediction": train_rows[chosen]["hi"]})
    return predictions
