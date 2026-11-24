"""Evaluate the frozen pretrained model on synthetic and optional local validation inputs."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import sacrebleu

from evaluation.contrast_scoring import load_contrasts, score_contrast
from saarthi_app.model import DEFAULT_MODEL_DIR, MODEL_ID, REVISION, Translator, INDIC_MODEL_DIR, INDIC_MODEL_ID, INDIC_REVISION, IndicTranslator

ROOT = Path(__file__).resolve().parent


def decode(translator, texts, beams, cap):
    outputs = []
    for offset in range(0, len(texts), 8):
        outputs.extend(translator.translate(
            texts[offset:offset + 8], beams=beams, max_new_tokens=cap, require_eos=False
        ))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--backend", choices=("opus", "indic"), default="indic")
    parser.add_argument("--val-en", type=Path)
    parser.add_argument("--val-hi", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.val_en) != bool(args.val_hi):
        parser.error("Supply both validation files or neither.")
    if args.output.exists():
        parser.error("Choose a new output file to preserve prior results.")

    if args.backend == "indic":
        translator = IndicTranslator(args.model_dir or INDIC_MODEL_DIR)
        model_id, revision = INDIC_MODEL_ID, INDIC_REVISION
        packages = ("ctranslate2", "bhashantar", "sentencepiece", "sacrebleu", "sacremoses")
    else:
        translator = Translator(args.model_dir or DEFAULT_MODEL_DIR)
        model_id, revision = MODEL_ID, REVISION
        packages = ("torch", "transformers", "sentencepiece", "sacrebleu", "sacremoses")
    smoke = json.loads((ROOT / "smoke.json").read_text(encoding="utf-8"))
    smoke_outputs = decode(translator, [row["text"] for row in smoke], beams=4, cap=192)
    contrasts = load_contrasts()
    contrast_outputs = decode(translator, [row["en"] for row in contrasts], beams=1, cap=64)
    scored = [
        {"id": row["id"], "category": row["category"], "prediction": prediction,
         **score_contrast(row, prediction)}
        for row, prediction in zip(contrasts, contrast_outputs, strict=True)
    ]
    report = {
        "model": model_id, "revision": revision,
        "software": {package: version(package) for package in packages},
        "display_check": {"beams": 4, "max_new_tokens": 192,
                          "examples": [dict(row, prediction=out) for row, out in zip(smoke, smoke_outputs, strict=True)]},
        "contrast_check": {"beams": 1, "max_new_tokens": 64,
                           "automatic_passes": sum(row["passed"] for row in scored),
                           "total": len(scored),
                           "automatic_by_category": dict(Counter(row["category"] for row in scored if row["passed"])),
                           "examples": scored},
    }

    if args.val_en:
        sources = args.val_en.read_text(encoding="utf-8").splitlines()
        refs = args.val_hi.read_text(encoding="utf-8").splitlines()
        if not (len(sources) == len(refs) == 520) or any(not text.strip() for text in sources + refs):
            parser.error("Expected the 520 aligned, nonempty validation pairs.")
        outputs = decode(translator, sources, beams=1, cap=64)
        bleu, chrf = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF(word_order=2)
        report["validation"] = {
            "pairs": len(refs), "beams": 1, "max_new_tokens": 64,
            "bleu": bleu.corpus_score(outputs, [refs]).score,
            "chrfpp": chrf.corpus_score(outputs, [refs]).score,
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
