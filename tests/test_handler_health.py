import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestHandlerHealth(unittest.TestCase):
    def test_health_check_does_not_use_runtime_services(self) -> None:
        registered = {}

        runpod = types.ModuleType("runpod")
        runpod.serverless = types.SimpleNamespace(
            start=lambda config: registered.update(config)
        )

        aiohttp = types.ModuleType("aiohttp")
        aiohttp.ClientSession = type("ClientSession", (), {})

        boto3 = types.ModuleType("boto3")
        botocore = types.ModuleType("botocore")
        botocore_config = types.ModuleType("botocore.config")
        botocore_config.Config = type("Config", (), {})

        redis = types.ModuleType("redis")
        redis_asyncio = types.ModuleType("redis.asyncio")
        redis_asyncio.from_url = lambda *args, **kwargs: object()
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
        spec = importlib.util.spec_from_file_location("handler_health_test", handler_path)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        response = asyncio.run(
            module.handler({"input": {"health_check": True}})
        )

        self.assertEqual(
            response,
            {"status": "healthy", "service": "ltx-2.5-worker"},
        )
        self.assertIs(registered["handler"], module.handler)


if __name__ == "__main__":
    unittest.main()
