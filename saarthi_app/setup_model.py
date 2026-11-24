"""Explicitly download just the pinned public demo files."""

import argparse
from pathlib import Path

from .model import (
    DEFAULT_MODEL_DIR, INDIC_FILES, INDIC_MODEL_DIR, INDIC_MODEL_ID, INDIC_REVISION,
    INDIC_WEIGHTS_SHA256, MODEL_FILES, MODEL_ID, OPUS_WEIGHTS_SHA256, REVISION, verify_weights,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opus", "indic"), default="indic")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    if args.backend == "indic":
        model_id, revision, files, default_dir = INDIC_MODEL_ID, INDIC_REVISION, INDIC_FILES, INDIC_MODEL_DIR
    else:
        model_id, revision, files, default_dir = MODEL_ID, REVISION, MODEL_FILES, DEFAULT_MODEL_DIR
    model_dir = args.model_dir or default_dir
    snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=list(files),
        local_dir=model_dir,
        token=False,
        max_workers=2,
    )
    missing = [name for name in files if not (model_dir / name).is_file()]
    if missing:
        parser.exit(1, "Model setup incomplete. Missing " + ", ".join(missing) + "\n")
    weights_name, digest = (
        ("model.bin", INDIC_WEIGHTS_SHA256)
        if args.backend == "indic"
        else ("pytorch_model.bin", OPUS_WEIGHTS_SHA256)
    )
    verify_weights(model_dir / weights_name, digest)
    print(f"Model files ready at {model_dir}")


if __name__ == "__main__":
    main()
