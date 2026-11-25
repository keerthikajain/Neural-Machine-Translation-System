# Independent bilingual review rubric

Use the existing 30 shuffled A/B items in `blind-review.json`. Model identities and the ordering key are absent from this packet. Review each output independently before comparing A and B. Judges should be proficient in English and Hindi and should record concrete evidence in the notes where possible. The packet is blank and no independent scores have been entered.

| Criterion | Mark pass when |
| --- | --- |
| Main action or state | The translation preserves what happened, is happening, or is requested. Material distinctions such as ability, intention, obligation, and actual occurrence remain intact. |
| Participants and objects | The same people or entities play the same roles. The direct object and any meaning-critical recipient remain identifiable. Hindi may drop explicitly recoverable pronouns. |
| Quantity and counted entity | The value, range, unit, and counted entity or event agree with the source. For instance, दो साल fails for two children even though दो matches. Accept equivalent forms such as साढ़े आठ for eight-thirty. |
| Negation | The same proposition is affirmed or denied, with the same scope. Judge a confirmation tag separately from the primary clause. |
| Overall factual adequacy | All applicable core facts pass and no material invented fact changes the meaning. Accept faithful paraphrases, natural Hindi word order, contextual formality choices, and minor errors that leave the intended meaning recoverable. |

Use `1` for pass, `0` for fail, and `NA` for an inapplicable field. Use `U` for source ambiguity that the reviewer cannot resolve and explain it. When model Hindi is opaque, mark a required fact as failed if it cannot be recovered. Do not infer a fact from the English alone. A minor morphology error can pass factual adequacy; a role, gender, number, or polarity change that alters a fact fails the relevant field.

The English is the meaning authority when a reference is inconsistent. Note the issue and consider all reference alternatives. Preserve the original item order and labels. The frozen protocol calls for at least **six net treatment wins** on the 30 paired overall judgments, calculated only after scoring is frozen and the separate key is revealed. Missing or unresolved overall judgments keep the independent gate pending. Report agreement and adjudication separately if two judges are available.
