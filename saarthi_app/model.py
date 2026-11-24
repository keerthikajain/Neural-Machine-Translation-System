"""CPU inference with the pinned publicly pretrained Saarthi demo backend."""

from hashlib import file_digest
from pathlib import Path
from threading import Lock

MODEL_ID = "Helsinki-NLP/opus-mt-en-hi"
REVISION = "75d7f7c9232b2891c7d65fe4ef635616c72be867"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "opus-mt-en-hi"
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
)
MAX_CHARS = 1200
MAX_SOURCE_TOKENS = 512
OPUS_WEIGHTS_SHA256 = "b07f16822d80aabd93e5ae05ac7e0dbda18989ecb3942e926214ef13ddb593dd"
INDIC_WEIGHTS_SHA256 = "c2cbf665f6a11655291641f6cf4bf047bd86bdec0c7ea368f948bc8a01b27c8d"


class ModelUnavailable(RuntimeError):
    pass


class InputTooLong(ValueError):
    pass


def verify_weights(path, expected):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ModelUnavailable("The local model checksum does not match the pinned weights.")


class Translator:
    def __init__(self, directory=DEFAULT_MODEL_DIR):
        self.directory = Path(directory)
        self._lock = Lock()
        self._loaded = None

    def _load(self):
        if any(not (self.directory / name).is_file() for name in MODEL_FILES):
            raise ModelUnavailable("The model files are missing. Run python -m saarthi_app.setup_model.")
        verify_weights(self.directory / "pytorch_model.bin", OPUS_WEIGHTS_SHA256)
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            torch.set_num_threads(min(4, torch.get_num_threads()))
            tokenizer = MarianTokenizer.from_pretrained(self.directory, local_files_only=True)
            model = MarianMTModel.from_pretrained(
                self.directory, local_files_only=True, use_safetensors=False, weights_only=True
            ).to("cpu").eval()
        except Exception as exc:
            raise ModelUnavailable("The local translation model could not load. Check the setup and restart Saarthi.") from exc
        self._loaded = torch, tokenizer, model

    def translate(self, texts, *, beams=4, max_new_tokens=192, require_eos=True):
        if not texts:
            return []
        with self._lock:
            if self._loaded is None:
                self._load()
            torch, tokenizer, model = self._loaded
            batch = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=False)
            source_limit = min(MAX_SOURCE_TOKENS, model.config.max_position_embeddings)
            if batch["input_ids"].shape[1] > source_limit:
                raise InputTooLong("The English text exceeds the model's token limit. Split it into smaller passages.")
            with torch.inference_mode():
                sequences = model.generate(
                    **batch, do_sample=False, num_beams=beams, max_new_tokens=max_new_tokens,
                    forced_eos_token_id=None,
                )
            eos = model.config.eos_token_id
            if require_eos and any(eos not in row[1:].tolist() for row in sequences):
                raise InputTooLong("The model could not finish translating this passage. Split it into smaller passages.")
            outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            if any(not text.strip() for text in outputs):
                raise ModelUnavailable("The model returned an empty translation.")
            return outputs


INDIC_MODEL_ID = "zaidwhys/indictrans2-en-indic-dist-200M-ct2-int8"
INDIC_REVISION = "abf8512c45febe9cc9fbb30ee781bb3be4f18956"
INDIC_MODEL_DIR = ROOT / "models" / "indictrans2-int8"
INDIC_FILES = (
    "config.json", "model.SRC", "model.TGT", "model.bin",
    "source_vocabulary.json", "target_vocabulary.json", "provenance.json", "LICENSE",
)


class IndicTranslator:
    def __init__(self, directory=INDIC_MODEL_DIR):
        self.directory = Path(directory)
        self._lock = Lock()
        self._loaded = None

    def _load(self):
        if any(not (self.directory / name).is_file() for name in INDIC_FILES):
            raise ModelUnavailable("The model files are missing. Run python -m saarthi_app.setup_model --backend indic.")
        verify_weights(self.directory / "model.bin", INDIC_WEIGHTS_SHA256)
        try:
            import ctranslate2
            import sentencepiece
            from bhashantar import textprep

            encoder = sentencepiece.SentencePieceProcessor(model_file=str(self.directory / "model.SRC"))
            model = ctranslate2.Translator(str(self.directory), device="cpu", compute_type="int8", intra_threads=4)
        except Exception as exc:
            raise ModelUnavailable("The local translation model could not load. Check the setup and restart Saarthi.") from exc
        self._loaded = textprep, encoder, model

    def translate(self, texts, *, beams=4, max_new_tokens=192, require_eos=True):
        if not texts:
            return []
        with self._lock:
            if self._loaded is None:
                self._load()
            textprep, encoder, model = self._loaded
            prepared = [textprep.prepare(text, "hin_Deva") for text in texts]
            sources = []
            for row in prepared:
                source_tag, target_tag, body = row.model_text.split(" ", 2)
                tokens = [source_tag, target_tag, *encoder.encode(body, out_type=str)]
                if len(tokens) + 1 > 256:
                    raise InputTooLong("The English text exceeds the model's token limit. Split it into smaller passages.")
                sources.append(tokens)
            result = model.translate_batch(
                sources, beam_size=beams, max_decoding_length=max_new_tokens, return_end_token=True,
            )
            outputs = []
            for row, item in zip(prepared, result, strict=True):
                pieces = item.hypotheses[0]
                if require_eos and "</s>" not in pieces:
                    raise InputTooLong("The model could not finish translating this passage. Split it into smaller passages.")
                decoded = "".join(piece for piece in pieces if piece not in {"<s>", "<pad>", "</s>", "<unk>"})
                output = textprep.finish(decoded.replace("▁", " ").strip(), "hin_Deva", row.placeholders)
                if not output.strip():
                    raise ModelUnavailable("The model returned an empty translation.")
                outputs.append(output)
            return outputs
