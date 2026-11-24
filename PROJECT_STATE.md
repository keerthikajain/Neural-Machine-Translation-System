# Saarthi V1 project state

## Ready for local demonstration

- The restored interface follows the Saarthi mockups with the original logo and purple skyline, a desktop sidebar, a Quick Phrases panel, and mobile bottom navigation.
- English → Hindi runs on CPU with the existing pinned IndicTrans2 INT8 model. The custom Saarthi Compact A remains separately documented as experimental.
- Quick Phrases fills the English field. History can be searched and cleared. Settings enables optional browser persistence for up to 30 translations.
- Research code, checkpoint lineage, and original artwork are preserved. The changes are local and uncommitted.

## Verification and open items

- All six local app tests passed with real-model testing enabled. Chromium exercised live desktop and mobile translations, keyboard navigation, copy, clear, error recovery, history, and settings. Layout checked at widths from 320 to 1440 pixels. Screenshots are in `docs/screenshots/`.
- Only English → Hindi is supported. Mockup extras such as the welcome screen, voice input, and language swapping are outside the V1 demo.
- Linux launcher verified. Windows release readiness requires running `run_demo.ps1` with installed dependencies on a Windows machine and translating a sample in the browser.
