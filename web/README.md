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

## Exact repository assets (no generated or substitute artwork)

| Reference role | Repository file | Used in |
|---|---|---|
| Welcome landscape | `public/illustrations/splash-landscape.png` | Welcome artwork and subtle result-card decoration |
| Desktop landmark landscape | `public/illustrations/desktop-landscape-fixed(1).png` | Desktop workspace bottom background |
| Current Saarthi brand logo | `public/illustrations/saarthi-logo.svg` | Welcome, desktop/mobile brand, browser icon |

**Reference mismatch requiring the original asset:** the checked-in SVG logo depicts overlapping English/Hindi speech bubbles (“A / अ”), while the supplied UI reference depicts a multicolored lotus. To follow the instruction to use exactly the checked-in assets, the interface deliberately keeps the original SVG rather than redrawing or replacing it. A pixel-accurate lotus match requires the actual lotus logo source file in the repository.

Screens and spacing target the provided reference; browser rendering, exact image cropping, fonts, and responsive breakpoints still need screenshot comparison before claiming 1:1 fidelity. No image file was changed.
