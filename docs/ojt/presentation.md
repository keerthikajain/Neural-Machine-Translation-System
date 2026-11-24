# Saarthi OJT presentation

**Main message.** I built an English–Hindi NMT research pipeline, measured where scratch-trained models failed, and integrated a pretrained model to deliver a working local translator.

| Slide | What to show and say |
|---|---|
| 1 · Problem and objective | Translate everyday English into usable Hindi. A useful translation should preserve the action, people, quantities, and instructions. |
| 2 · NMT in one minute | Neural machine translation models the probability of the target sentence conditioned on the source. The encoder represents English pieces. The decoder generates Hindi pieces one at a time and attends to the English representation. |
| 3 · Why start from scratch? | Implementing tokenization, masking, attention, optimization, checkpointing, and evaluation made each stage inspectable. Explain [Compact A’s architecture and workflow](compact-a.md). |
| 4 · Scratch-training challenge | Corpus alignment, limited GPU time, repetitive outputs, and factual errors. Show the Compact A repaired-corpus table, including the 0/42 contrast result. Distinguish falling validation loss from reliable generation. |
| 5 · Why Mini? | A smaller trainable model makes matched CPU experiments practical. Show the [Mini workflow](mini.md) and explain what stayed fixed in Pilot 3. |
| 6 · What we learned | Present 13.45 → 16.14 chrF++, the missed gain threshold, and 0/11 reliable quantity–referent outputs. Describe why a number token alone is weak evidence and what the coverage audit can establish. |
| 7 · Why use IndicTrans2? | Attribute the pretrained weights to AI4Bharat and the INT8 conversion separately. Show [V1’s inference diagram](v1-app.md). My work here was integrating and verifying the application. |
| 8 · Live demo | Run V1 locally. Translate the train-journey example, point out both cities and the duration, click Copy, and open History. Choose a Quick Phrase and translate it. |
| 9 · Limits and next work | Show the three-component comparison. Mention the pretrained evaluation overlap uncertainty and decimal error. Explain the Windows smoke test and independent Mini review gates. |

## Demo checklist

1. Use the [V1 guide](v1-app.md) to install the model before the presentation. Rehearse with the exact checkout you will show.
2. Start the local server and run the train example once so the model is warm.
3. Keep the [published screenshot](https://raw.githubusercontent.com/keerthikajain/Neural-Machine-Translation-System/f706501da5be18e126c636d5bfe92b34e669cef5/docs/screenshots/desktop.jpg) and saved output ready in case the live environment fails. Label the screenshot as recorded evidence if you use it.
4. End by pointing to the three component guides and the [verification report](verification.md).

The implemented deliverable is the V1 English → Hindi demo. Pilot 4 treatment training, an independent Mini result, and V1 integration of either custom model remain future decisions.
