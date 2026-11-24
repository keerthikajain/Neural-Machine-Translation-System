# Saarthi

Saarthi translates English text into Hindi on your device. The working demo uses a pinned, pretrained **AI4Bharat IndicTrans2 distilled 200M** model with a CPU INT8 conversion. The separately trained **Saarthi Compact A** Transformer remains an experimental model in this repository.

![Saarthi translating an English passage into Hindi](docs/screenshots/desktop.jpg)

[Mobile layout screenshot](docs/screenshots/mobile.jpg)

The purple dashboard uses Saarthi’s original logo and skyline artwork, with a desktop sidebar and mobile bottom navigation. Quick Phrases fills the English input. Translation History is searchable in English and Hindi. It lasts until reload by default, and Settings can save up to 30 translations in the current browser.

## Run locally

Use Python 3.11 or 3.12 on a supported 64-bit CPU, plus Node.js 20 or newer for the React build. Install the pinned Python dependencies in a virtual environment. The one-time model setup downloads roughly 224 MB. Startup and inference use local files and require no API account.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m saarthi_app.setup_model
npm ci
npm run build
python -m saarthi_app
```

### Windows PowerShell

From the repository root, using Python 3.12 and Node.js 20 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
npm.cmd ci
npm.cmd run build
.\.venv\Scripts\python.exe -m saarthi_app.setup_model
$env:SAARTHI_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
.\run_demo.ps1
```

If the Python launcher `py` is missing, check that `python --version` reports Python 3.11 or 3.12, then use `python -m venv .venv`. If PowerShell blocks the script, start the same server with `.\.venv\Scripts\python.exe -m saarthi_app`. The Windows launcher still needs an end-to-end smoke test on Windows.

Open [localhost:8000](http://127.0.0.1:8000). The first translation loads the model and may take longer. On Linux or macOS you can also start the server with `./run_demo.sh` after activating the environment. Use `--port 8080` if port 8000 is in use. Ctrl+C stops the server.

Try **Anita took the train from Pune to Mumbai. The journey lasted 3 hours.** Saarthi produced **अनीता ने पुणे से मुंबई के लिए ट्रेन ली। यह यात्रा 3 घंटे तक चली।**

## Evaluation

| Model | BLEU ↑ | chrF++ ↑ | Automated source contrasts ↑ |
|---|---:|---:|---:|
| Custom Saarthi 20K | 0.345 | 10.956 | 0/42 |
| Custom Saarthi 3K pilot best | 0.267 | 11.826 | 0/42 |
| Custom Saarthi 5K pilot latest | 0.288 | 11.406 | 0/42 |
| Helsinki-NLP OPUS-MT | 6.166 | 26.822 | 13/42 |
| IndicTrans2 distilled 200M INT8 | **23.210** | **49.674** | **34/42** |

Scores use the same 520 validation rows, greedy decoding with a 64-token cap, and SacreBLEU 2.6.0. Each model has its own tokenizer. A separate check of the interactive decoding settings found **18/20** examples preserved all reviewed facts. IndicTrans2 once translated 4.2 km as 4 km on the greedy synthetic contrast. Review names, numbers and instructions before reuse. Public pretraining may overlap with the validation set, so validation metrics do not establish an independent held-out benchmark.

See [evaluation and reproduction](docs/evaluation.md), [architecture and model provenance](docs/architecture.md), [demo plan](docs/demo.md), and [project state](PROJECT_STATE.md).

## Tests

Build the React bundle with `npm run build` before running the app tests.

```bash
python -m unittest discover -s tests -p 'test_local_app.py' -v
# Include the real-model test once setup is complete
SAARTHI_TEST_MODEL=1 python -m unittest discover -s tests -p 'test_local_app.py' -v
```

In PowerShell set `$env:SAARTHI_TEST_MODEL = '1'` before running the test command.

The React source is in `saarthi_app/app.jsx`. Run `npm run build` after changing it. The Python server serves the built bundle and handles translation requests. The original research implementation, training utilities, and diagnostics remain in `src/`, `notebooks/`, and the root training scripts. Model weights, private datasets, and private evaluation outputs are excluded from Git.

See the [original research README](https://github.com/keerthikajain/Neural-Machine-Translation-System/blob/febc34af7ab5ef63322f57df40a61fb73818185c/README.md) for the training walkthrough and research project structure.
