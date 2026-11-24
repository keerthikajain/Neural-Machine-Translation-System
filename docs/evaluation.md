# Reproduce the evaluation

Install the [app dependencies](../README.md) and set up the model. The synthetic suites are versioned in [smoke.json](../evaluation/smoke.json) and [contrasts.json](../evaluation/contrasts.json). The same 520 aligned validation lines are supplied locally. The evaluator writes metrics and synthetic predictions, with no validation text in the output file.

```bash
python -m evaluation.run_translation --backend indic \
  --val-en /path/to/val_en.txt --val-hi /path/to/val_hi.txt \
  --output evaluation/private/indic_run.json
```

For the synthetic-only check, omit both validation flags. To reproduce the OPUS-MT comparison in the same environment, also install the optional CPU dependencies and set up that model separately.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.53.3
python -m saarthi_app.setup_model --backend opus
```

Use `--backend opus` in the evaluation command with a new output path. Both backends use greedy decoding capped at 64 tokens for comparison and beam 4 with a 192-token cap for the 20 app examples.

## Results and limits

- [IndicTrans2 frozen output and metrics](../evaluation/indic_results.json) · [OPUS-MT frozen output and metrics](../evaluation/opus_results.json)
- [Manual review notes](../evaluation/manual_review.json) contain the rubric, failures, and contrast scorer disagreements.
- IndicTrans2 preserved the contrast and its core context in **41/42** synthetic cases by one reviewer's assessment. The 34/42 automatic count is the conservative result under the fixed keyword and context rules used for all model comparisons. The decimal degradation from 4.2 km to 4 km is a genuine failure. The human review also documents minor wording caveats.
- The existing Saarthi numbers came from the private 20K screened-corpus pilot, version 1. The 50K checkpoint remains historical. It was excluded from the working translator and was not rerun for this delivery.
- BLEU uses `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. chrF++ uses `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`.
- The single-reviewer synthetic check is small. Possible overlap between public training and validation corpora was not assessed. Private validation files and checkpoint binaries remain outside Git.
