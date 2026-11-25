# Pilot 4 matched data intervention protocol

This freezes the go/no-go design for the existing proposal. It does not authorize a run. The experiment requires the 30-item independent review to be completed and archived, 120 approved replacements, a frozen and approved 60-item diagnostic, verified development overlap, and a digest-bound clearance from an authorized steward for the sealed partitions. A completed Pilot 3 review is evidence, irrespective of whether its model met the Pilot 3 gain threshold. Only a pass/fail attestation is exchanged for the sealed partitions. The audit and builder have no path that opens those examples.

## Matched arms

Start both from the verified update-600 checkpoint and restore the same optimizer buffers and RNG. Use the original tokenizer, architecture, 0.0001 continuation rate, ordering seed and shuffling procedure, batch size 32, loss, gradient clipping, and 40-token greedy decoder. The control uses all 2,397 original rows. Treatment replaces precisely 120 rows at the frozen indices, with 60 action–role and 60 quantity–referent examples. Retain all other records. There is no corpus expansion by row count.

Use two CPU threads, 300 elapsed seconds, 768 MiB RSS, and a maximum of 1,050 updates per arm. Evaluate every 150 updates. Baseline validation NLL at update 600 is 3.003686478586191. Stop at two consecutive validation NLL values strictly above baseline plus 0.3, three consecutive checks without a full-precision 0.5 chrF++ gain over the last accepted best, or any update, elapsed-time, memory, or numerical limit. Compare at the treatment's latest accepted best on the shared completed checkpoint horizon, against control at the identical update. No new accepted best means no advancement. Archive both arms and recovery state separately.

Replacement may change length, number of scored tokens per batch, vocabulary, style and reference diversity even at the same positions and row count. Any effect is the result of replacement as a package. It cannot establish the isolated effects of cleaner references, new words, author style or semantic coverage. Use the prospective and actual distribution reports to assess those differences.

## Frozen evaluation

Before training, a bilingual team should create 60 development-only diagnostic sources, 30 primarily testing action/agent/object or recipient roles and 30 primarily testing quantities and counted entities or units. Use source constructions distinct from both training versions and the original validation. Deduplicate by the existing two-language family checks and independent paraphrase review. Balance positive, negative, question and request constructions while keeping the source natural. Use two independent reviewers, with adjudication before model identities are revealed. Record main action, agent/object, quantity with referent, negation scope and overall factual adequacy. Preserve blinded scoring and the source/target review ledger.

| Gate at the matched checkpoint | Required improvement |
| --- | --- |
| Overall factual adequacy | At least +6 net wins across 60 paired items |
| Action and participant/object roles | At least +3 net wins across the 30 action items, requiring both event and roles |
| Quantity and counted entity/unit | At least +3 net wins across the 30 quantity items, requiring both value and referent |
| Negation | Among frozen diagnostic items tagged with explicit negation, require no net reduction in correctly scoped negation. Create at least 10 such items with distinct families. |
| Original validation chrF++ guardrail | At least control minus 0.5 |

Net wins equal treatment-only correct minus control-only correct. Require complete adjudicated relevant labels for each gate. Report wins, losses, ties, percentage-point differences, two-sided exact McNemar p-values (binomial test on discordant pairs) and 95% paired bootstrap percentile intervals with 10,000 resamples, seed 4042026. Resample pairs within the frozen strata for overall adequacy and within the applicable subset for secondary gates. Report p-values and intervals as uncertainty, without a separate significance gate or tuning on any diagnostic outputs. Report original validation BLEU, chrF++, exact matches, repetition, EOS, names and the lexical number proxy. These descriptive metrics do not replace factual gates. A passing pilot is evidence for the intervention under its tested conditions. It does not authorize GPU training, held-out inference or V1 integration.
