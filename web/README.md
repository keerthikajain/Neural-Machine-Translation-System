# Saarthi frontend preview

This standalone, responsive UI is based on the supplied mobile/desktop reference and reuses the existing lotus logo and landscape images in `public/illustrations`.

## Run locally

From the **repository root**:

```powershell
python -m http.server 5173
```

Open `http://localhost:5173/web/`.

## What works

- Mobile welcome page, responsive workspace and desktop sidebar
- English-to-Hindi language display, text entry, clearing and character counter
- Built-in **phrasebook** example translations, copy, speech playback (browser-dependent)
- Searchable local phrasebook history, quick phrases and basic navigation
- Browser speech recognition when supported

## What is intentionally not implemented

- **No actual neural-model translation yet.** Free text explains that a trained checkpoint and inference API are required; it never fabricates output.
- The language direction is fixed to English → Hindi because only that model is planned.
- No user authentication or server-backed history.
- Not production hardened or verified against all browsers.

## Safety and integration

The frontend has no build dependencies, does not alter backend, training, model, or data files, and is isolated in `web/`. The UI branch must be reviewed and tested before merging. Once trained model weights and an inference service exist, wire the Translate action to a documented, validated API; do not replace phrasebook examples with guessed translations.

## Design reference

Pastel lavender/pink palette, lotus branding, welcome landscape, mobile tab navigation and desktop sidebar were taken from the provided Saarthi image. The two existing landscape PNGs and SVG logo are loaded from the repository. Fonts and proportions are approximations; compare rendered screenshots with the reference before declaring a pixel-match.
