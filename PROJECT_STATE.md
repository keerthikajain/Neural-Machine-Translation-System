# Saarthi V1 project state

## Ready for local demonstration

- The React interface preserves the original logo and translucent purple skyline, with a desktop sidebar, one Quick Phrases page, and mobile bottom navigation.
- English → Hindi runs on CPU with the existing pinned IndicTrans2 INT8 model. The custom Saarthi Compact A remains separately documented as experimental.
- Quick Phrases fills the English field. History can be searched and cleared. Settings enables optional browser persistence for up to 30 translations.
- The restoration is committed on the local `feature/saarthi-v1-react-verified` branch, based on draft PR #5. Build the frontend before startup using the [README](README.md).

## Verification and open items

- All six local app tests passed with real-model testing enabled. Chromium exercised live desktop and mobile translations, keyboard navigation, copy, clear, error recovery, history, and settings. Layout checked at widths from 320 to 1440 pixels. Screenshots are in `docs/screenshots/`.
- Only English → Hindi is supported. Mockup extras such as the welcome screen, voice input, and language swapping are outside the V1 demo.
- Linux launcher verified. Windows release readiness requires running `run_demo.ps1` with installed dependencies on a Windows machine and translating a sample in the browser.
