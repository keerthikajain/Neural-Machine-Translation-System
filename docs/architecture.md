# Architecture and model provenance

The browser sends English text to the local Python server at `POST /api/translate`. The server validates the body, loads the demo model on the first request, translates on the CPU, then returns Hindi as JSON. The browser displays the response as text and can copy it to the clipboard. The application never needs an inference API or the training corpus.

| Role | Model and provenance | Execution |
|---|---|---|
| Working translator | AI4Bharat IndicTrans2 En→Indic Distilled 200M. The demo uses the MIT-licensed CTranslate2 INT8 conversion by zaidwhys, revision `abf8512c45febe9cc9fbb30ee781bb3be4f18956`. | CTranslate2 on CPU, `eng_Latn → hin_Deva`, beam 4 in the app. |
| Pretrained comparison | Helsinki-NLP OPUS-MT En→Hi, Apache-2.0, revision `75d7f7c9232b2891c7d65fe4ef635616c72be867`. | Marian PyTorch on CPU, kept in the evaluation harness. |
| Project research | Saarthi Compact A, our 31,980,800-parameter Transformer encoder–decoder and 32K SentencePiece V2 vocabulary. The 20K, 50K and pilot checkpoints belong to this lineage. | Separate training and evaluation scripts. The repaired-corpus pilot received a NO-GO. |

The demo uses an explicit model setup step and verifies the model weight file by SHA-256 at load time. It accepts up to 1,200 characters and enforces the model's 256-position source limit after tokenization. If generation fails to terminate within its cap, the app asks the user to split the passage. It does not silently clip the input.

The custom-model effort covers the original IIT Bombay corpus audit, partial repairs, privacy-screened release, and checkpoint diagnostics. The demo does not load or distribute those checkpoints or data.

[AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) · [Pinned INT8 conversion](https://huggingface.co/zaidwhys/indictrans2-en-indic-dist-200M-ct2-int8/tree/abf8512c45febe9cc9fbb30ee781bb3be4f18956) · [Helsinki-NLP OPUS-MT](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
