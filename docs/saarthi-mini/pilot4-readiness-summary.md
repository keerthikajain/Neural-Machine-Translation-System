# Saarthi Mini Pilot 4 readiness

**NO-GO for training.** Mini is a scratch-trained English-to-Hindi research Transformer. Pilot 3 increased matched validation chrF++ from 13.45 to 16.14, while exact matches stayed at 0/203. The earlier analyst audit found that none of 11 numeric examples preserved both the value and the thing counted. The user also reports an exploratory human-assisted check where both outputs failed on all 20 reviewed items. That check is separate from the pending independent 30-item review and supplies no independent scores.

## Inventory and preservation

| Item | Verified state |
| --- | --- |
| Baseline | Public repository base `febc34a`. The local Pilot 1–4 lineage includes `63ddea0`, `82c0c85`, `c4553ac`, `8e3cbcc`, and `b137682`. |
| Local preparation | `saarthi-mini-pilot4-curation` points to `b137682`. A separate readiness branch holds the new tooling. The existing untracked contrast file stayed untouched. |
| Datasets | Verified original manifest and unchanged 2,397 train rows and 216 validation rows representing 203 validation inputs. All 120 proposed replacements remain pending. |
| Models | Update-600 full recovery state, selected weights and tokenizer match prior hashes. Pilot 1, 2, 3 and the historical 50K artifact were preserved. |
| Evaluation | Frozen 30-item packet, all A/B assignments and row order verified. Separate 60-item development diagnostic is drafted and screened. No scoring or inference took place. |
| Private materials | Corpus rows, replacement manifest, diagnostic and contrast examples, review packets, assignment keys and model weights reside outside the public source backup. |

## Readiness checklist

- [x] Verify previous work, parent checkpoints, development hashes and frozen A/B worksheet.
- [x] Prepare reviewer handoff, candidate flags and a 120-row blank bilingual worksheet.
- [x] Draft 30 action and 30 quantity diagnostic items and apply normalized and two-language family checks to development inputs.
- [ ] Collect two genuinely independent A/B worksheets, archive their hashes and adjudicate differences without revealing the arm key.
- [ ] Obtain two recorded bilingual approvals per replacement and owner licensing clearance for all 120 pairs. Edits require fresh pair hashes and approval.
- [ ] Have the authorized review team verify and freeze the separate 60-item diagnostic, including reference scope, semantic tags, paraphrase separation and provenance.
- [ ] Obtain the steward's digest-bound sealed-overlap attestation, without opening sealed material in the research workflow.
- [ ] After clearance, build and hash the treatment dataset. Implement and dry-audit a Pilot 4 paired runner before authorizing training. The existing Pilot 3 runner compares learning rates and binds the original dataset fingerprint. It cannot serve as the treatment runner unchanged.

## Evaluation and reproducibility

The original 30-item packet has no assignment key, model names or scores. The coordinator keeps reviewer identities separate from worksheets provided to the blinded adjudicator. Reviewers score each side against the English, consulting all permitted Hindi references. The adjudicator reviews unresolved disagreements while blinded, records reasons, and freezes the final worksheets before the mapping is released. Unresolved required labels remain pending. This design stores the raw and adjudicated receipts separately. Reviewer independence and semantic approval require actual people. Neither the draft nor automated checks create that evidence.

Training data, original validation, the proposed 60-item diagnostic, and the independent held-out items play distinct roles. The diagnostic is reserved for a single post-selection comparison between arms. The first draft's radio-message item collided with a development family and was replaced. The final automated screen found no conflicts with original training, validation, or proposed replacements and no cross-item family collisions. Human paraphrase review remains necessary. The 60 drafts include 12 scoped-negation items in distinct families, eight named-entity items, 30 explicitly object-tagged items, 11 question and 12 request items, and 12 items tagged for unsupported-content sensitivity. Missing arguments and fluent wrong meanings are rated against the source facts. No model outputs exist for this draft.

The matched design is unchanged. Both arms must restore the approved update-600 checkpoint's model, optimizer and RNG, use seed 2417 and the existing batch permutation, batch size 32, sequence limit 64, original 1,600-piece tokenizer, constant 0.0001 continuation rate, and 40-token greedy decoding. Preserve the full-precision shared-checkpoint selection rule and original validation scorer. Keep optimizer history and runtime fingerprints in each arm's manifest. Verify actual update numbers and batch index order with a dry preflight after the dataset is approved. Pilot 4 scoring checks complete adjudication and frozen-ID order. It implements paired net wins, a two-sided exact McNemar calculation and seeded paired bootstrap intervals on the frozen strata.

Required thresholds are +6/60 factual-adequacy net wins, +3/30 action–role wins, +3/30 quantity–referent wins, no net scoped-negation loss, and validation chrF++ at least control minus 0.5. BLEU, chrF++, exact match, repetition and EOS remain descriptive outside the existing selection and guardrail. See the unchanged [frozen design](pilot4-frozen-design.md).

## Technical priorities and OJT explanation

1. **Meaning preservation.** The largest observed errors are dropped actions and objects and invented facts. Prioritize independent adequacy and role/quantity judgments over lexical accuracy.
2. **Coverage and reference quality.** The training audit found limited co-occurrence coverage. Flag scope, role, register and measurement mismatches for bilingual review. Treat the replacement policy as a package because vocabulary, style and length can all shift.
3. **Training reproducibility.** Make parent checkpoint identity and ordering observable. Preserve original validation bytes and require a verified change manifest for the treatment. Keep both arms isolated from the application.
4. **Small evaluation samples.** Report paired uncertainty and do not treat an exploratory review or overlap-screen pass as evidence of reliable generalization.

OJT summary: “I built a small English-to-Hindi Transformer from scratch and ran controlled experiments to compare how training changes affect its output. The scores improved, but bilingual inspection showed that the model still drops important parts of sentences and can attach numbers to the wrong thing. I documented those failures and prepared a blinded review and a controlled dataset comparison. Saarthi Mini remains experimental while the separate V1 translator uses IndicTrans2.”
