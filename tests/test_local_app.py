import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from saarthi_app.__main__ import create_server
from saarthi_app.model import INDIC_FILES, IndicTranslator, InputTooLong, ModelUnavailable


class FixtureTranslator:
    def translate(self, texts):
        text, = texts
        if text == "missing":
            raise ModelUnavailable("Model files are missing.")
        if text == "over limit":
            raise InputTooLong("Split the passage.")
        if text == "crash":
            raise RuntimeError("internal detail")
        return ["मीरा ने रेडियो ठीक किया।"]


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(("127.0.0.1", 0), FixtureTranslator())
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, data, content_type="application/json"):
        request = Request(self.url + "/api/translate", data, {"Content-Type": content_type})
        try:
            response = urlopen(request, timeout=4)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_assets_and_route_allowlist(self):
        for path in ("/", "/logo.svg", "/styles.css", "/app.js", "/skyline-desktop.png", "/skyline-mobile.png"):
            with urlopen(self.url + path, timeout=4) as response:
                self.assertTrue(response.read())
        with self.assertRaises(HTTPError) as failure:
            urlopen(self.url + "/README.md", timeout=4)
        self.assertEqual(failure.exception.code, 404)

    def test_success_and_unicode(self):
        status, body = self.post(b'{"text":" Meera fixed the radio. "}')
        self.assertEqual(status, 200)
        self.assertEqual(body["translation"], "मीरा ने रेडियो ठीक किया।")

    def test_bad_input(self):
        for body in (b'{', b'{"text": 1}', b'{"text":" "}', b'[]', br'{"text":"\ud800"}'):
            with self.subTest(body=body):
                self.assertEqual(self.post(body)[0], 400)
        self.assertEqual(self.post(b'{"text":"hello"}', "text/plain")[0], 415)
        self.assertEqual(self.post(json.dumps({"text": "a" * 1201}).encode())[0], 413)

    def test_model_failures(self):
        for text, expected in (("missing", 503), ("over limit", 422), ("crash", 500)):
            status, body = self.post(json.dumps({"text": text}).encode())
            self.assertEqual(status, expected)
            self.assertNotIn("internal detail", body["error"])


class ModelLoadTests(unittest.TestCase):
    def test_refuses_corrupt_weights_before_loading_backend(self):
        with TemporaryDirectory() as directory:
            translator = IndicTranslator(directory)
            with self.assertRaisesRegex(ModelUnavailable, "missing"):
                translator.translate(["Hello"])
            for filename in INDIC_FILES:
                (Path(directory) / filename).touch()
            with self.assertRaisesRegex(ModelUnavailable, "checksum"):
                translator.translate(["Hello"])


@unittest.skipUnless(os.getenv("SAARTHI_TEST_MODEL") == "1", "Requires installed demo weights")
class RealModelTest(unittest.TestCase):
    def test_local_model_is_deterministic_and_preserves_count(self):
        translator = IndicTranslator()
        inputs = ["Exactly 18 students passed the exam."]
        first = translator.translate(inputs)
        self.assertEqual(first, translator.translate(inputs))
        self.assertRegex(first[0], r"18|१८|अठारह")
        self.assertRegex(first[0], r"छात्र|विद्यार्थ")
