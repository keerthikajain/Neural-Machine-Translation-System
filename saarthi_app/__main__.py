"""Serve the Saarthi demo on a local HTTP address."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path

from .model import INDIC_MODEL_DIR, MAX_CHARS, ROOT, IndicTranslator, InputTooLong, ModelUnavailable

MAX_BODY = 16_384
FRONTEND_BUNDLE = "saarthi_app/static/app.js"
FRONTEND_BUILD_HINT = "Build the React frontend from the project root with npm ci, then npm run build."
ASSETS = {
    "/": ("saarthi_app/index.html", "text/html; charset=utf-8"),
    "/app.js": (FRONTEND_BUNDLE, "text/javascript; charset=utf-8"),
    "/styles.css": ("saarthi_app/styles.css", "text/css; charset=utf-8"),
    "/skyline-desktop.png": ("public/illustrations/desktop-landscape-fixed(1).png", "image/png"),
    "/skyline-mobile.png": ("public/illustrations/splash-landscape.png", "image/png"),
    "/logo.svg": ("public/illustrations/saarthi-logo.svg", "image/svg+xml"),
}


def create_server(address, translator):
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, status, kind, payload):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if kind.startswith("application/json"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, status, **fields):
            self.send_payload(
                status, "application/json; charset=utf-8",
                json.dumps(fields, ensure_ascii=False).encode("utf-8"),
            )

        def do_GET(self):
            asset = ASSETS.get(self.path)
            if asset is None:
                self.send_json(404, error="Page not found.")
                return
            path, kind = asset
            if path == FRONTEND_BUNDLE and not (ROOT / path).is_file():
                self.send_json(503, error=FRONTEND_BUILD_HINT)
                return
            self.send_payload(200, kind, (ROOT / path).read_bytes())

        def do_POST(self):
            if self.path != "/api/translate":
                self.send_json(404, error="Endpoint not found.")
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, error="Send a JSON request.")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length > MAX_BODY:
                self.send_json(413, error="The request is too large.")
                return
            if length <= 0:
                self.send_json(400, error="Send a valid JSON request.")
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError):
                self.send_json(400, error="Send a valid JSON request.")
                return
            raw = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                self.send_json(400, error="Enter English text to translate.")
                return
            text = raw.strip()
            if len(text) > MAX_CHARS:
                self.send_json(413, error=f"Enter at most {MAX_CHARS:,} characters.")
                return
            try:
                text.encode("utf-8")
                translation, = translator.translate([text])
            except UnicodeEncodeError:
                self.send_json(400, error="The text contains an invalid Unicode character.")
            except InputTooLong as exc:
                self.send_json(422, error=str(exc))
            except ModelUnavailable as exc:
                self.send_json(503, error=str(exc))
            except Exception:
                logging.error("Translation inference failed")
                self.send_json(500, error="Translation failed. Try again with a shorter passage.")
            else:
                self.send_json(200, translation=translation)

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model-dir", default=INDIC_MODEL_DIR, type=Path)
    args = parser.parse_args()
    if not (ROOT / FRONTEND_BUNDLE).is_file():
        parser.error(FRONTEND_BUILD_HINT)
    with create_server((args.host, args.port), IndicTranslator(args.model_dir)) as server:
        print(f"Saarthi is running at http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
