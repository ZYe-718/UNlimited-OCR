import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import infer


class LoopbackSessionTest(unittest.TestCase):
    def test_loopback_session_does_not_inherit_proxy_environment(self):
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.example:8080",
                "HTTPS_PROXY": "http://proxy.example:8080",
                "ALL_PROXY": "http://proxy.example:8080",
            },
            clear=False,
        ):
            settings = infer.LOOPBACK_SESSION.merge_environment_settings(
                "http://127.0.0.1:10000/health", {}, None, None, None
            )

        self.assertFalse(infer.LOOPBACK_SESSION.trust_env)
        self.assertEqual(settings["proxies"], {})

    def test_server_ready_uses_loopback_session(self):
        response = SimpleNamespace(status_code=200)
        with patch.object(infer.LOOPBACK_SESSION, "get", return_value=response) as get:
            self.assertTrue(infer.server_ready("http://127.0.0.1:10000"))

        get.assert_called_once_with("http://127.0.0.1:10000/health", timeout=5)

    def test_infer_one_uses_loopback_session(self):
        response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        args = SimpleNamespace(image_mode="gundam")
        original_ngram_size = infer.NO_REPEAT_NGRAM_SIZE
        infer.NO_REPEAT_NGRAM_SIZE = 0
        try:
            with (
                patch.object(infer.LOOPBACK_SESSION, "post", return_value=response) as post,
                patch.object(infer, "build_content", return_value=[]),
                patch.object(
                    infer,
                    "collect_stream_silent",
                    return_value={"tokens": 1, "decode_time": 0, "text": "ok"},
                ),
            ):
                result = infer.infer_one("sample.png", None, args, 1)
        finally:
            infer.NO_REPEAT_NGRAM_SIZE = original_ngram_size

        self.assertEqual(result["text"], "ok")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:10000/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()