import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import frontend_app
from tests.test_frontend_app import FakeClientSession, FakeResponse
from workflow_support import build_output_path, safe_input_path, write_input_images


class TestSecurityBoundaries(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(frontend_app.app)

    def test_submit_rejects_unapproved_urls_without_network_access(self):
        urls = (
            "http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8000/runsync",
            "https://evil.example/runsync", "https://api.runpod.ai.evil.example/v2/id/run",
            "https://api.runpod.ai@evil.example/v2/id/run",
            "https://api.runpod.ai/v2/../run", "https://api.runpod.ai/v2/%2e%2e/run",
            "https://api.runpod.ai/v2/id/run?redirect=http://127.0.0.1",
            "https://api.runpod.ai/v2/id/run#fragment",
            "https://api.runpod.ai:9000/v2/id/run", "https://api.runpod.ai\\@evil.example",
            "https://api.runpod.ai/v2/id/ru\nn", "http://[::1]/runsync",
        )
        with patch.object(frontend_app, "FRONTEND_SUBMIT_URLS", ()):
            for url in urls:
                with self.subTest(url=url), patch.object(
                    frontend_app.aiohttp, "ClientSession"
                ) as session:
                    response = self.client.post("/api/submit", json={
                        "endpoint_url": url, "payload": {"input": {}}
                    })
                    self.assertEqual(response.status_code, 422)
                    session.assert_not_called()

    def test_submit_preserves_runpod_and_exact_configured_urls_without_redirects(self):
        custom = "http://127.0.0.1:8000/runsync"
        urls = ("https://api.runpod.ai/v2/test-id/runsync",
                "https://api.runpod.ai/v2/test-id/run", custom)
        with patch.object(frontend_app, "FRONTEND_SUBMIT_URLS", (custom,)):
            for url in urls:
                session = FakeClientSession(post_response=FakeResponse(
                    status=200, json_data={"id": "job-123"}
                ))
                with self.subTest(url=url), patch.object(
                    frontend_app.aiohttp, "ClientSession", return_value=session
                ):
                    response = self.client.post("/api/submit", json={
                        "endpoint_url": url, "auth_token": "test-token",
                        "payload": {"input": {}}
                    })
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json()["ok"])
                    self.assertEqual(response.json()["response_json"], {"id": "job-123"})
                    self.assertEqual(session.requests, [(url, {
                        "headers": {"Content-Type": "application/json",
                                    "Authorization": "Bearer test-token"},
                        "allow_redirects": False,
                    })])
            for suffix in ("/../admin", "?url=http://169.254.169.254", "#fragment"):
                response = self.client.post("/api/submit", json={
                    "endpoint_url": custom + suffix, "payload": {}
                })
                self.assertEqual(response.status_code, 422)

    def test_polling_rejects_unconfigured_nodes_without_network_access(self):
        for node in ("169.254.169.254", "evil.example", "127.0.0.1:9000",
                     "127.0.0.1:8188@evil.example", "127.0.0.1:8188/path"):
            with self.subTest(node=node), patch.object(
                frontend_app.aiohttp, "ClientSession"
            ) as session:
                response = self.client.get(
                    "/api/pod-submit/prompt-123", params={"node": node}
                )
                self.assertEqual(response.status_code, 400)
                session.assert_not_called()

    def test_polling_uses_configured_node_without_redirects(self):
        session = FakeClientSession(get_response=FakeResponse(status=200))
        with patch.object(frontend_app, "LOCAL_COMFY_NODE", "comfy.internal:8188"), \
                patch.object(frontend_app.aiohttp, "ClientSession", return_value=session):
            response = self.client.get("/api/pod-submit/prompt-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.requests, [
            ("http://comfy.internal:8188/history/prompt-123", {"allow_redirects": False})
        ])

    def test_polling_accepts_equivalent_configured_node_representations(self):
        for node in ("http://127.0.0.1:8188", "127.0.0.1:8188/", " "):
            session = FakeClientSession(get_response=FakeResponse(status=200))
            with self.subTest(node=node), patch.object(
                frontend_app, "LOCAL_COMFY_NODE", "127.0.0.1:8188"
            ), patch.object(frontend_app.aiohttp, "ClientSession", return_value=session):
                response = self.client.get(
                    "/api/pod-submit/prompt-123", params={"node": node}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(session.requests[0][0],
                                 "http://127.0.0.1:8188/history/prompt-123")

    def test_polling_rejects_prompt_url_metacharacters(self):
        for prompt_id in ("bad%3Fquery", "bad%23fragment", "bad%25escape", "bad%5Cpath"):
            with self.subTest(prompt_id=prompt_id), patch.object(
                frontend_app.aiohttp, "ClientSession"
            ) as session:
                self.assertEqual(
                    self.client.get(f"/api/pod-submit/{prompt_id}").status_code, 400
                )
                session.assert_not_called()

    def test_input_and_output_reject_outside_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "files"
            root.mkdir()
            outside = Path(tmp) / "files-other"
            outside.mkdir()
            victim = outside / "secret.txt"
            victim.write_text("untouched")
            (root / "link").symlink_to(outside, target_is_directory=True)
            for name in ("../files-other/secret.txt", str(victim), "link/secret.txt"):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        safe_input_path(str(root), name)
                    with self.assertRaises(ValueError):
                        write_input_images(str(root), [{"name": name, "image": "eA=="}])
                    with self.assertRaises(ValueError):
                        build_output_path(str(root), {"filename": name})
            for subfolder in ("../files-other", str(outside), "link"):
                with self.subTest(subfolder=subfolder), self.assertRaises(ValueError):
                    build_output_path(str(root), {
                        "filename": "secret.txt", "subfolder": subfolder
                    })
            self.assertEqual(victim.read_text(), "untouched")

    def test_output_route_enforces_real_boundary_and_serves_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "output"
            (root / "nested").mkdir(parents=True)
            (root / "nested" / "clip.mp4").write_bytes(b"video")
            secret = Path(tmp) / "secret.txt"
            secret.write_text("secret")
            (root / "link").symlink_to(secret)
            with patch.dict(os.environ, {"COMFY_OUTPUT_DIR": str(root)}):
                for name in ("../secret.txt", str(secret), "link"):
                    with self.subTest(name=name):
                        response = self.client.get(
                            "/api/comfy-output", params={"filename": name}
                        )
                        self.assertEqual(response.status_code, 400)
                response = self.client.get("/api/comfy-output", params={
                    "filename": "clip.mp4", "subfolder": "nested", "download": True
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"video")
                self.assertIn("attachment", response.headers["content-disposition"])

    def test_cleanup_preserves_outside_files_and_removes_nested_uploads(self):
        # Load the worker cleanup without starting RunPod or its runtime services.
        source = Path(frontend_app.__file__).with_name("handler.py").read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "cleanup_input_files")
        namespace = {"Path": Path, "os": os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "handler.py", "exec"),
             namespace)
        for cleanup in (frontend_app.cleanup_input_files, namespace["cleanup_input_files"]):
            with self.subTest(cleanup=cleanup), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "input"
                nested = root / "job"
                nested.mkdir(parents=True)
                upload = nested / "source.png"
                upload.write_bytes(b"image")
                outside = Path(tmp) / "input-other"
                outside.mkdir()
                victim = outside / "secret.txt"
                victim.write_text("untouched")
                (root / "link").symlink_to(outside, target_is_directory=True)
                namespace["COMFY_INPUT_DIR"] = str(root)
                with patch.object(frontend_app, "COMFY_INPUT_DIR", str(root)):
                    cleanup([str(victim), str(root / "link" / "secret.txt"), str(upload)])
                self.assertEqual(victim.read_text(), "untouched")
                self.assertFalse(nested.exists())
                self.assertTrue(root.is_dir())


if __name__ == "__main__":
    unittest.main()
