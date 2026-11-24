# Saarthi viva notes

## A 45-second explanation

> Saarthi is an English-to-Hindi translation project. I implemented and trained a custom Transformer with a parallel-data pipeline, SentencePiece, checkpointing, and evaluation. I then built a smaller version to run controlled CPU experiments. Higher learning rates improved some text-overlap scores, but the model still lost important actions and quantities. For the working demo, I integrated AI4Bharat’s pretrained IndicTrans2 model into my Saarthi interface. The demo runs locally on CPU. The research findings explain what I learned about training and why I chose that deployment path.

| Question | Short answer |
|---|---|
| What is neural machine translation? | A neural model generates a sentence in the target language conditioned on a sentence in the source language. |
| Why an encoder and a decoder? | The encoder represents the English input. The decoder generates Hindi from its previous output and attends to those English representations. |
| What does SentencePiece do? | It maps text into subword IDs, which lets the model represent many words with a fixed vocabulary. Compact A and Mini use different learned vocabularies. |
| Why does low validation loss not guarantee good translations? | Teacher forcing supplies the correct earlier tokens. During generation, the model conditions on its own predictions, and mistakes can compound. |
| What did Pilot 3 control? | Both arms resumed the same weights, optimizer state, and random state. Architecture, data, and decoding settings were held fixed. Only the continuation learning rate differed. |
| What happened in Pilot 3? | chrF++ rose from 13.45 to 16.14, short of the +3.0 gate. Three number strings matched, but none of the 11 quantity cases reliably preserved both the value and what it counted. |
| What did the audit prove? | It documented specific coverage gaps and meaning errors. The contribution of a single cause is still unresolved, which is why the next intervention needs independent review and a controlled design. |
| What model translates in the demo? | AI4Bharat’s pretrained IndicTrans2, using a pinned INT8 conversion. I built the application and inference integration. |
| How would you improve the research? | First complete bilingual review and data-curation clearance. Then evaluate a controlled intervention against action, role, quantity, and negation criteria. |
| What should I avoid claiming? | Keep the claims tied to the measured development sets and the observed demo. Use the [comparison and limitations](../../README.md#what-we-measured) when discussing results. |

For code and report references, see [Compact A](compact-a.md), [Mini](mini.md), and [V1](v1-app.md).
