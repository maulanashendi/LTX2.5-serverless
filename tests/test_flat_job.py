import asyncio
import base64
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
TINY_PNG_DATA_URL = f"data:image/png;base64,{TINY_PNG_BASE64}"

FAKE_HISTORY = {"outputs": {}}


class FakeClientTimeout:
    def __init__(self, *args, **kwargs):
        pass


class FakeClientSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class TestFlatJob(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def load_handler(self):
        runpod = types.ModuleType("runpod")
        runpod.serverless = types.SimpleNamespace(start=lambda config: None)

        aiohttp = types.ModuleType("aiohttp")
        aiohttp.ClientSession = FakeClientSession
        aiohttp.ClientTimeout = FakeClientTimeout

        boto3 = types.ModuleType("boto3")
        botocore = types.ModuleType("botocore")
        botocore_config = types.ModuleType("botocore.config")
        botocore_config.Config = type("Config", (), {})

        redis = types.ModuleType("redis")
        redis_asyncio = types.ModuleType("redis.asyncio")
        redis_asyncio.from_url = Mock(return_value=Mock())
        redis.asyncio = redis_asyncio

        modules = {
            "aiohttp": aiohttp,
            "boto3": boto3,
            "botocore": botocore,
            "botocore.config": botocore_config,
            "redis": redis,
            "redis.asyncio": redis_asyncio,
            "runpod": runpod,
        }

        handler_path = Path(__file__).resolve().parents[1] / "handler.py"
        spec = importlib.util.spec_from_file_location("handler_flat_job_test", handler_path)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop("REDIS_URL", None)
            spec.loader.exec_module(module)

        module.COMFY_INPUT_DIR = self.tmpdir.name
        return module

    def stub_execution(self, module, history=None, side_effect=None):
        module.execute_workflow = AsyncMock(
            return_value=history if history is not None else FAKE_HISTORY,
            side_effect=side_effect,
        )
        module.build_workflow_output_payload = AsyncMock(
            return_value={"videos": [{"filename": "out.mp4", "subfolder": "", "type": "base64",
                                       "data": "AA==", "media_type": "video/mp4"}]}
        )

    def run_handler(self, module, job_input, job_id="job-1"):
        return asyncio.run(module.handler({"id": job_id, "input": job_input}))

    # 1. prompt only -> t2v, no node "395"
    def test_prompt_only_selects_t2v(self):
        module = self.load_handler()
        self.stub_execution(module)

        response = self.run_handler(module, {"prompt": "a cat"})

        self.assertEqual(response["status"], "success")
        graph = module.execute_workflow.call_args.args[1]
        self.assertNotIn("395", graph)

    # 2. prompt + data-URL image -> i2v, node "395" scoped by job id
    def test_prompt_with_data_url_image_selects_i2v(self):
        module = self.load_handler()
        self.stub_execution(module)

        response = self.run_handler(
            module,
            {"prompt": "a cat", "image": TINY_PNG_DATA_URL, "image_name": "cat.png"},
            job_id="job-2",
        )

        self.assertEqual(response["status"], "success")
        graph = module.execute_workflow.call_args.args[1]
        self.assertEqual(graph["395"]["inputs"]["image"], "job-2/cat.png")

    # 3. raw base64 image (no data: prefix) -> same as above
    def test_prompt_with_raw_base64_image_selects_i2v(self):
        module = self.load_handler()
        self.stub_execution(module)

        response = self.run_handler(
            module,
            {"prompt": "a cat", "image": TINY_PNG_BASE64, "image_name": "cat.png"},
            job_id="job-3",
        )

        self.assertEqual(response["status"], "success")
        graph = module.execute_workflow.call_args.args[1]
        self.assertEqual(graph["395"]["inputs"]["image"], "job-3/cat.png")

    # 4. health_check short-circuits before touching workflow execution
    def test_health_check_never_executes_workflow(self):
        module = self.load_handler()
        module.execute_workflow = AsyncMock(side_effect=AssertionError("must not run"))

        response = self.run_handler(module, {"health_check": True})

        self.assertEqual(response, {"status": "healthy", "service": "ltx-2.5-worker"})
        module.execute_workflow.assert_not_called()

    # 5. workflow key wins over prompt
    def test_workflow_key_wins_over_prompt(self):
        module = self.load_handler()
        self.stub_execution(module)
        module.build_graph = Mock(side_effect=AssertionError("must not build a graph"))

        raw_workflow = {"1": {"class_type": "Whatever", "inputs": {}}}
        response = self.run_handler(
            module, {"prompt": "a cat", "workflow": raw_workflow}
        )

        self.assertEqual(response["status"], "success")
        sent_graph = module.execute_workflow.call_args.args[1]
        self.assertEqual(sent_graph, raw_workflow)

    # 6. missing/blank prompt raises, does not return an error dict
    def test_missing_or_blank_prompt_raises(self):
        module = self.load_handler()
        self.stub_execution(module)

        with self.assertRaises(ValueError):
            self.run_handler(module, {})
        with self.assertRaises(ValueError):
            self.run_handler(module, {"prompt": "   "})

    # 7. duration + aspect ratio feed through to the graph
    def test_duration_and_aspect_ratio_set_graph_nodes(self):
        module = self.load_handler()
        self.stub_execution(module)

        self.run_handler(module, {"prompt": "a cat", "duration": 7, "aspect_ratio": "9:16"})

        graph = module.execute_workflow.call_args.args[1]
        self.assertEqual(graph["398:362"]["inputs"]["value"], 7)
        self.assertEqual(graph["398:372"]["inputs"]["value"], 720)
        self.assertEqual(graph["398:360"]["inputs"]["value"], 1280)

    # 8. same seed -> identical graphs across calls
    def test_seed_makes_graph_deterministic(self):
        module = self.load_handler()
        self.stub_execution(module)

        self.run_handler(module, {"prompt": "a cat", "seed": 123}, job_id="job-a")
        graph_1 = module.execute_workflow.call_args.args[1]

        self.run_handler(module, {"prompt": "a cat", "seed": 123}, job_id="job-b")
        graph_2 = module.execute_workflow.call_args.args[1]

        self.assertEqual(graph_1, graph_2)

    # 9. explicit mode conflicting with image presence raises
    def test_mode_conflicting_with_image_presence_raises(self):
        module = self.load_handler()
        self.stub_execution(module)

        with self.assertRaises(ValueError):
            self.run_handler(
                module, {"prompt": "a cat", "image": TINY_PNG_DATA_URL, "mode": "t2v"}
            )
        with self.assertRaises(ValueError):
            self.run_handler(module, {"prompt": "a cat", "mode": "i2v"})

    # 10. exceptions from workflow execution propagate out of handler()
    def test_execution_failure_propagates(self):
        module = self.load_handler()
        self.stub_execution(module, side_effect=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.run_handler(module, {"prompt": "a cat"})

    # 11. input files are cleaned up in `finally`, even on failure
    def test_input_files_cleaned_up_even_on_failure(self):
        module = self.load_handler()
        self.stub_execution(module, side_effect=RuntimeError("boom"))

        job_id = "job-cleanup"
        with self.assertRaises(RuntimeError):
            self.run_handler(
                module,
                {"prompt": "a cat", "image": TINY_PNG_DATA_URL, "image_name": "cat.png"},
                job_id=job_id,
            )

        job_dir = Path(self.tmpdir.name) / job_id
        self.assertFalse(job_dir.exists())

    # 12. an 'api_key' field is ignored, job still succeeds
    def test_api_key_field_is_ignored(self):
        module = self.load_handler()
        self.stub_execution(module)

        response = self.run_handler(module, {"prompt": "a cat", "api_key": "anything"})

        self.assertEqual(response["status"], "success")

    # 13. flat-path and workflow-path envelopes share the same structure
    def test_envelope_structure_matches_workflow_path(self):
        module = self.load_handler()
        self.stub_execution(module)

        flat_response = self.run_handler(module, {"prompt": "a cat"}, job_id="job-flat")
        workflow_response = self.run_handler(
            module,
            {"workflow": {"1": {"class_type": "Whatever", "inputs": {}}}},
            job_id="job-workflow",
        )

        self.assertEqual(set(flat_response), set(workflow_response))
        self.assertEqual(set(flat_response["metadata"]), set(workflow_response["metadata"]))


if __name__ == "__main__":
    unittest.main()
