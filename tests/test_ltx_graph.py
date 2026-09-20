from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import ltx_graph
from ltx_graph import build_graph
from workflow_support import ALLOW_REMOTE_IMAGE_ENV, MAX_IMAGE_BYTES, materialize_image


class FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, size: int):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, status: int, chunks: list[bytes]) -> None:
        self.status = status
        self.content = FakeContent(chunks)


class FakeGetContext:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> FakeResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, bool]] = []

    def get(self, url: str, allow_redirects: bool = True):
        self.calls.append((url, allow_redirects))
        return FakeGetContext(self.response)


class NoNetworkSession:
    def get(self, *args, **kwargs):
        raise AssertionError("materialize_image must not issue a network request here")


class TestBuildGraphModes(unittest.TestCase):
    def test_t2v_has_no_image_node(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot")
        self.assertNotIn("395", graph)

    def test_i2v_sets_image_node(self) -> None:
        graph = build_graph(mode="i2v", prompt="a drone shot", image_name="scene.png")
        self.assertEqual(graph["395"]["inputs"]["image"], "scene.png")

    def test_node_counts(self) -> None:
        t2v = build_graph(mode="t2v", prompt="a drone shot")
        i2v = build_graph(mode="i2v", prompt="a drone shot", image_name="scene.png")
        self.assertEqual(len(t2v), 42)
        self.assertEqual(len(i2v), 48)


class TestBuildGraphDefaults(unittest.TestCase):
    def test_default_duration(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot")
        self.assertEqual(graph["398:362"]["inputs"]["value"], 5)

    def test_default_aspect_ratio(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot")
        self.assertEqual(graph["398:372"]["inputs"]["value"], 1280)
        self.assertEqual(graph["398:360"]["inputs"]["value"], 720)

    def test_fps_always_24(self) -> None:
        for aspect_ratio in ("16:9", "9:16", "1:1"):
            graph = build_graph(mode="t2v", prompt="a drone shot", aspect_ratio=aspect_ratio)
            self.assertEqual(graph["398:361"]["inputs"]["value"], 24)


class TestBuildGraphAspectRatios(unittest.TestCase):
    def test_9_16(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot", aspect_ratio="9:16")
        self.assertEqual(graph["398:372"]["inputs"]["value"], 720)
        self.assertEqual(graph["398:360"]["inputs"]["value"], 1280)

    def test_1_1(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot", aspect_ratio="1:1")
        self.assertEqual(graph["398:372"]["inputs"]["value"], 1024)
        self.assertEqual(graph["398:360"]["inputs"]["value"], 1024)

    def test_unsupported_aspect_ratio(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="a drone shot", aspect_ratio="4:3")


class TestBuildGraphDuration(unittest.TestCase):
    def test_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="a drone shot", seconds=0)

    def test_above_max_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="a drone shot", seconds=21)

    def test_fractional_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="a drone shot", seconds=5.5)

    def test_boundaries_accepted(self) -> None:
        build_graph(mode="t2v", prompt="a drone shot", seconds=1)
        build_graph(mode="t2v", prompt="a drone shot", seconds=20)


class TestBuildGraphPromptOptimizer(unittest.TestCase):
    def test_disabled(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot", optimize_prompt=False)
        self.assertEqual(graph["398:380"]["inputs"]["sampling_mode"], "off")
        self.assertIs(graph["398:383"]["inputs"]["value"], False)

    def test_enabled(self) -> None:
        graph = build_graph(mode="t2v", prompt="a drone shot", optimize_prompt=True)
        self.assertEqual(graph["398:380"]["inputs"]["sampling_mode"], "on")
        self.assertIs(graph["398:383"]["inputs"]["value"], True)

    def test_sampling_mode_seed_always_int(self) -> None:
        for optimize_prompt in (True, False):
            graph = build_graph(
                mode="t2v", prompt="a drone shot", optimize_prompt=optimize_prompt
            )
            self.assertIsInstance(graph["398:380"]["inputs"]["sampling_mode.seed"], int)


class TestBuildGraphSeed(unittest.TestCase):
    def test_same_seed_is_deterministic(self) -> None:
        first = build_graph(mode="t2v", prompt="a drone shot", seed=12345)
        second = build_graph(mode="t2v", prompt="a drone shot", seed=12345)
        self.assertEqual(first["398:338"], second["398:338"])
        self.assertEqual(first["398:339"], second["398:339"])
        self.assertEqual(first, second)

    def test_without_seed_calls_differ(self) -> None:
        first = build_graph(mode="t2v", prompt="a drone shot")
        second = build_graph(mode="t2v", prompt="a drone shot")
        self.assertNotEqual(
            first["398:338"]["inputs"]["noise_seed"],
            second["398:338"]["inputs"]["noise_seed"],
        )


class TestBuildGraphValidation(unittest.TestCase):
    def test_blank_prompt_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="")

    def test_whitespace_prompt_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt="   ")

    def test_missing_prompt_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="t2v", prompt=None)  # type: ignore[arg-type]

    def test_unsupported_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="v2v", prompt="a drone shot")

    def test_i2v_requires_image_name(self) -> None:
        with self.assertRaises(ValueError):
            build_graph(mode="i2v", prompt="a drone shot")


class TestBuildGraphDoesNotMutateTemplate(unittest.TestCase):
    def test_template_untouched_across_calls(self) -> None:
        before_t2v = copy.deepcopy(ltx_graph._TEMPLATES["t2v"])
        before_i2v = copy.deepcopy(ltx_graph._TEMPLATES["i2v"])

        build_graph(mode="t2v", prompt="first call", seconds=3, aspect_ratio="9:16")
        build_graph(mode="i2v", prompt="second call", image_name="a.png", seed=99)

        self.assertEqual(ltx_graph._TEMPLATES["t2v"], before_t2v)
        self.assertEqual(ltx_graph._TEMPLATES["i2v"], before_i2v)


class TestMaterializeImage(unittest.IsolatedAsyncioTestCase):
    async def test_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scoped, path = await materialize_image(
                tmp, "job-1", "data:image/png;base64,dGVzdA==", "scene.png"
            )
            self.assertEqual(scoped, "job-1/scene.png")
            self.assertEqual(Path(path).read_bytes(), b"test")

    async def test_raw_base64(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scoped, path = await materialize_image(tmp, "job-1", "dGVzdA==", "scene.png")
            self.assertEqual(scoped, "job-1/scene.png")
            self.assertEqual(Path(path).read_bytes(), b"test")

    async def test_http_without_env_var_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, ALLOW_REMOTE_IMAGE_ENV):
                await materialize_image(
                    tmp,
                    "job-1",
                    "https://example.com/image.png",
                    "scene.png",
                    session=NoNetworkSession(),
                )

    async def test_http_allowed_downloads_successfully(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(FakeResponse(200, [b"remote-bytes"]))
            with patch.dict(os.environ, {ALLOW_REMOTE_IMAGE_ENV: "true"}):
                scoped, path = await materialize_image(
                    tmp,
                    "job-1",
                    "https://example.com/image.png",
                    "scene.png",
                    session=session,
                )
            self.assertEqual(scoped, "job-1/scene.png")
            self.assertEqual(Path(path).read_bytes(), b"remote-bytes")
            self.assertEqual(session.calls, [("https://example.com/image.png", False)])

    async def test_http_redirect_rejected(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(FakeResponse(302, [b""]))
            with patch.dict(os.environ, {ALLOW_REMOTE_IMAGE_ENV: "true"}):
                with self.assertRaises(ValueError):
                    await materialize_image(
                        tmp,
                        "job-1",
                        "https://example.com/image.png",
                        "scene.png",
                        session=session,
                    )

    async def test_http_private_host_rejected_before_request(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {ALLOW_REMOTE_IMAGE_ENV: "true"}):
                for host in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "localhost"):
                    with self.subTest(host=host):
                        with self.assertRaises(ValueError):
                            await materialize_image(
                                tmp,
                                "job-1",
                                f"http://{host}/image.png",
                                "scene.png",
                                session=NoNetworkSession(),
                            )

    async def test_http_body_over_max_bytes_rejected(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            oversized_chunk = b"x" * (MAX_IMAGE_BYTES + 1)
            session = FakeSession(FakeResponse(200, [oversized_chunk]))
            with patch.dict(os.environ, {ALLOW_REMOTE_IMAGE_ENV: "true"}):
                with self.assertRaises(ValueError):
                    await materialize_image(
                        tmp,
                        "job-1",
                        "https://example.com/image.png",
                        "scene.png",
                        session=session,
                    )

    async def test_image_name_traversal_is_sanitized_and_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scoped, path = await materialize_image(
                tmp, "job-1", "dGVzdA==", "../../etc/passwd"
            )
            self.assertNotIn("..", scoped)
            resolved_path = Path(path).resolve()
            job_dir = (Path(tmp) / "job-1").resolve()
            self.assertEqual(resolved_path.parent, job_dir)
            self.assertTrue(str(resolved_path).startswith(str(job_dir)))
            self.assertFalse((Path(tmp) / "etc").exists())


if __name__ == "__main__":
    unittest.main()
