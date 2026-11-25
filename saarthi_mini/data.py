"""Build Saarthi Mini's local, versioned dataset from an attributed export."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import re
import unicodedata
from zipfile import ZipFile

VERSION = "mini-tatoeba-20260213-v1"
ARCHIVE_SHA = "6078908ae9d9f67535d8e0cb8314cc84e350940c62afc09b8a690d67c674ff08"
ATTRIBUTION = re.compile(
    r"CC-BY 2\.0 \(France\) Attribution: tatoeba\.org #(\d+) \(([^()]+)\) & #(\d+) \(([^()]+)\)"
)
SENSITIVE = re.compile(
    r"\b(?:password|passcode|secret|credential|token|e-?mail|address|telephone|"
    r"phone number|credit card|pin code|passport|account|login|social security)\b"
    r"|पासवर्ड|फ़ोन नंबर|फोन नंबर|ईमेल|पता है मेरा|@|https?://|\d{4,}",
    re.IGNORECASE,
)
EN = re.compile(r"[A-Za-z0-9 .,!?:;'’\-]+")
HI = re.compile(r"[\u0900-\u097f0-9 .,!?:;'’\-]+")
NUMBERS = set("zero one two three four five six seven eight nine ten eleven twelve".split())
PRONOUNS = set("i you he she we they me him her us them my your his our their".split())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"[\u0966-\u096f]", lambda m: str(unicodedata.decimal(m[0])), text)
    return " ".join(re.sub(r"[^\w\u0900-\u0963\u0966-\u097f']", " ", text).split())


def filter_reason(en, hi):
    if SENSITIVE.search(en) or SENSITIVE.search(hi):
        return "privacy_pattern"
    if not EN.fullmatch(en) or not HI.fullmatch(hi) or not re.search(r"[\u0904-\u0939]", hi):
        return "encoding_or_script"
    en_tokens, hi_tokens = canonical(en).split(), canonical(hi).split()
    if not 3 <= len(en_tokens) <= 12 or not 2 <= len(hi_tokens) <= 18:
        return "length"
    if not en.endswith((".", "?", "!")) or en.count("?") != hi.count("?"):
        return "sentence_boundary_or_question_mismatch"
    if en.endswith("!") or en.count(".") + en.count("?") != 1:
        return "outside_domain"
    if not 0.55 <= len(hi_tokens) / len(en_tokens) <= 2:
        return "length_ratio"
    if set(re.findall(r"\d+", canonical(en))) != set(re.findall(r"\d+", canonical(hi))):
        return "digit_mismatch"
    return None


def load(archive):
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("Source archive does not match the pinned 2026-02-13 export")
    rows, rejected = [], []
    seen = set()
    with ZipFile(archive) as source:
        for number, line in enumerate(source.read("hin.txt").decode("utf-8-sig").splitlines(), 1):
            en, hi, attribution = line.split("\t")
            match = ATTRIBUTION.fullmatch(attribution)
            if not match:
                raise ValueError(f"Missing sentence-level attribution at row {number}")
            en_id, en_author, hi_id, hi_author = match.groups()
            row_id = f"{en_id}:{hi_id}"
            reason = filter_reason(en, hi)
            key = canonical(en), canonical(hi)
            if key in seen:
                reason = reason or "duplicate_pair"
            if reason:
                rejected.append({"id": row_id, "source_row": number, "reason": reason})
                continue
            seen.add(key)
            rows.append({
                "id": row_id, "en": en, "hi": hi, "source_row": number,
                "en_author": en_author, "hi_author": hi_author,
                "attribution": attribution,
            })
    return rows, rejected


def families(rows, languages=("en", "hi")):
    """Connect variants transitively on either language before splitting."""
    parent = list(range(len(rows)))

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        parent[root(left)] = root(right)

    for language in languages:
        signatures = {}
        char_index = defaultdict(list)
        for index, row in enumerate(rows):
            text = canonical(row[language])
            words = tuple(text.split())
            number_words = tuple("<num>" if word.isdecimal() or
                                 (language == "en" and word in NUMBERS) else word
                                 for word in words)
            patterns = {words, number_words}
            # One missing or replaced word, plus two-slot variants for longer templates.
            if len(words) >= 3:
                patterns.update(words[:i] + words[i + 1:] for i in range(len(words)))
            if len(words) >= 6:
                patterns.update(tuple(word if i not in slots else "*" for i, word in enumerate(words))
                                for slots in combinations(range(len(words)), 2))
            if language == "en":
                patterns.add(tuple("<person>" if w in PRONOUNS else w for w in number_words))
                # Subject/number variants with a second changed slot.
                normalized = tuple("<person>" if w in PRONOUNS else w for w in number_words)
                if len(words) >= 5:
                    patterns.update(normalized[:i] + normalized[i + 1:] for i in range(len(words)))
            for pattern in patterns:
                if pattern in signatures:
                    join(signatures[pattern], index)
                signatures[pattern] = index
            grams = {text[i:i + 4] for i in range(len(text) - 3)}
            candidates = Counter(other for gram in grams for other in char_index[gram])
            for other, common in candidates.items():
                previous = canonical(rows[other][language])
                if common >= 0.5 * min(len(grams), len(previous) - 3):
                    similarity = SequenceMatcher(None, text, previous, autojunk=False)
                    if similarity.quick_ratio() >= 0.88 and similarity.ratio() >= 0.88:
                        join(other, index)
            for gram in grams:
                char_index[gram].append(index)

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[root(index)].append(row)
    return sorted(grouped.values(), key=lambda group: min(row["id"] for row in group))


def build(archive, output, review_path):
    rejected_ids = set(json.loads(review_path.read_text(encoding="utf-8"))["rejected_ids"])
    rows, rejected = load(archive)
    groups = families(rows)
    random.Random(9041).shuffle(groups)
    splits = {"validation": [], "test": [], "train": []}
    for group in groups:
        family_id = min(row["id"] for row in group)
        destination = min(("validation", "test"), key=lambda split: len(splits[split]))
        if len(splits[destination]) >= 220 or len(group) > 35:
            destination = "train"
        splits[destination].extend(dict(row, family=family_id) for row in group)
    for split, entries in splits.items():
        for row in entries:
            if row["id"] in rejected_ids:
                rejected.append({"id": row["id"], "source_row": row["source_row"],
                                 "reason": "bilingual_review"})
        splits[split] = [row for row in entries if row["id"] not in rejected_ids]
    output.mkdir(parents=True, exist_ok=False)
    for split, entries in splits.items():
        write_rows(output / f"{split}.jsonl", sorted(entries, key=lambda row: row["source_row"]))
    write_rows(output / "quarantine.jsonl", rejected)
    manifest = {
        "version": VERSION, "source_sha256": ARCHIVE_SHA,
        "source_url": "https://www.manythings.org/anki/hin-eng.zip",
        "license": "CC BY 2.0 France", "rejection_review_sha256": sha(review_path),
        "splits": {
            split: {"pairs": len(entries), "families": len({row["family"] for row in entries}),
                    "questions": sum(row["en"].endswith("?") for row in entries),
                    "sha256": sha(output / f"{split}.jsonl")}
            for split, entries in splits.items()
        },
        "largest_family": max(map(len, groups)),
        "quarantine_counts": dict(Counter(row["reason"] for row in rejected)),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--review", type=Path, default=Path(__file__).with_name("review.json"))
    args = parser.parse_args()
    build(args.archive, args.output, args.review)


if __name__ == "__main__":
    main()
