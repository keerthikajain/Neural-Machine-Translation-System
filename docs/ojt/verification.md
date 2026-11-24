# Repository and presentation verification

This review used separate worktrees for the documentation and the published V1 app. The main research checkout, Mini branches, and uncommitted React revision were kept intact.

| Evidence or check | Outcome |
|---|---|
| Repository status | `main` at `febc34a`. PR #2 merged Compact A. PRs #3, #4, and #5 were open drafts. #3 is separate training optimization. #4 is a superseded phrasebook preview. #5 contains the working V1 inference demo. |
| Compact A imports and entry points | Six data, model, training, and evaluation modules imported. The preprocessing and training CLIs returned help. The legacy launcher reported its missing local corpus and tokenizer prerequisites. A fresh Compact A instance reported 31,980,800 parameters. |
| Compact A focused tests | 31 passed on CPU across the model/training, dataset, and Compact A suites. Full discovery encountered a pre-existing import failure in `test_phase4_experiments.py` because `experiments.tiny_overfit` is absent from `main`. |
| Mini tests and CLI | 12 local Mini tests passed at `b137682`. Six entry points returned help, covering the original trainer, Pilots 2 and 3, decoding diagnostics, the coverage audit, and curation. |
| Published V1 | Six tests passed at `f706501` with real model testing enabled. The Linux launcher returned help and the local HTTP endpoint translated the presentation sample with the names, cities, and three-hour duration intact. |
| Local React V1 | Six app tests passed with the real model enabled. A previously recorded Chromium run exercised desktop and mobile translation, clipboard, history and persistence, Quick Phrases, error recovery, and widths from 320 to 1440 px. The local screenshot pair matched that browser run byte-for-byte. |
| Screenshots and links | README embeds the desktop screenshot and links the mobile screenshot from the pinned GitHub V1 revision. The local React revision is identified separately. Setup paths, diagram references, and existing source and screenshot paths were checked. |
| Remaining verification | An end-to-end run of `run_demo.ps1` on Windows is pending. The historical training runs were not repeated. Mini’s independent evaluation remains sealed. |

The documentation diff is limited to the repository README and the six OJT guides. It contains aggregate results and a synthetic demo sentence. There are no model binaries, dataset rows, keys, or sealed evaluation payloads. Test fixtures used CPU only. No new training experiment was started, and no model or app files changed.

[Compact A](compact-a.md) · [Mini](mini.md) · [V1 demo](v1-app.md) · [Presentation](presentation.md) · [Viva](viva.md)
