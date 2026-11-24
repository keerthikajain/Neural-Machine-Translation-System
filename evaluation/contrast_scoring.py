"""Source-fidelity rules from the custom Saarthi V2 pilot."""

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent


def canon(text):
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in unicodedata.normalize("NFC", text))
    return " ".join(text.casefold().split())


def contains(text, value):
    value = canon(value)
    if value[0].isdigit():
        return bool(re.search(r"(?<![\d.])" + re.escape(value) + r"(?!\d|\.\d)", text))
    # Devanagari combining marks are part of the surrounding word.
    return bool(re.search(r"(?<![\w\u0900-\u0963\u0972-\u097f])" + re.escape(value)
                          + r"(?![\w\u0900-\u0963\u0972-\u097f])", text))


def score_contrast(row, prediction):
    text = canon(prediction)
    opposing = any(contains(text, x) for x in row["opposing"])
    expected = any(contains(text, x) for x in row["expected"])
    # An affirmative keyword under an explicit negative does not establish meaning.
    negated = row["category"] == "meaning" and any(contains(text, x) for x in ("नहीं", "नही"))
    context = all(any(contains(text, x) for x in group) for group in row["context"])
    return dict(expected_found=expected, opposing_found=opposing, negation_flag=negated,
                context_found=context, passed=expected and context and not opposing and not negated)


def load_contrasts():
    rows = json.loads((ROOT / "contrasts.json").read_text(encoding="utf-8"))
    assert Counter(r["category"] for r in rows) == {"name": 14, "number": 14, "meaning": 14}
    assert len({r["en"] for r in rows}) == len({r["id"] for r in rows}) == 42
    assert all(score_contrast(r, r["hi"])["passed"] for r in rows)
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        assert left["group"] == right["group"]
        assert not score_contrast(left, right["hi"])["passed"]
        assert not score_contrast(right, left["hi"])["passed"]
    return rows
