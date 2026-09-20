import asyncio
import importlib.util
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


class TestHandlerHealth(unittest.TestCase):
    def load_handler(self, redis_url=None):
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
        self.redis_factory = Mock(return_value=object())
        redis_asyncio.from_url = self.redis_factory
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

        with patch.dict(sys.modules, modules), patch.dict(os.environ):
            os.environ.pop("REDIS_URL", None)
            if redis_url is not None:
                os.environ["REDIS_URL"] = redis_url
            spec.loader.exec_module(module)
        return module, registered

    def test_health_check_does_not_use_runtime_services(self) -> None:
        module, registered = self.load_handler()
        response = asyncio.run(
            module.handler({"input": {"health_check": True}})
        )

        self.assertEqual(
            response,
            {"status": "healthy", "service": "ltx-2.5-worker"},
        )
        self.assertIs(registered["handler"], module.handler)

    def test_redis_is_local_for_worker_and_hub_startup(self):
        for hub_validation in ("false", "true"):
            for url in (None, "redis://127.0.0.1:6379", "redis://localhost:6379",
                        "redis://localhost:6379/", "redis://127.0.0.1:6379/0"):
                with self.subTest(url=url, hub=hub_validation), patch.dict(
                    os.environ, {"RUNPOD_HUB_VALIDATION": hub_validation}
                ):
                    self.load_handler(url)
                    self.assertEqual(self.redis_factory.call_args.args,
                                     ("redis://127.0.0.1:6379",))

    def test_external_redis_rejected_before_client_creation(self):
        for url in self.unsafe_redis_urls:
            with self.subTest(url=url), self.assertRaisesRegex(
                RuntimeError, "External Redis is disabled"
            ) as error:
                self.load_handler(url)
            self.redis_factory.assert_not_called()
            self.assertNotIn(url, str(error.exception))

    unsafe_redis_urls = (
        "redis://external.example:6379", "rediss://external.example:6379",
        "redis://localhost.evil.example:6379", "redis://127.0.0.1.evil.example:6379",
        "redis://127.0.0.1:6379@external.example:6379",
        "redis://user:secret@external.example:6379",
        "redis://127.0.0.1:6379?host=external.example", "unix:///tmp/redis.sock",
        "redis://127.0.0.1:6380", "redis://127.0.0.1:6379/1",
    )

    def test_local_redis_startup_rejects_external_urls_without_commands(self):
        source = (Path(__file__).resolve().parents[1] / "src/start.sh").read_text()
        function = source.split("start_local_redis() {", 1)[1].split("\n}\n", 1)[0]
        for url in self.unsafe_redis_urls:
            result = subprocess.run(
                ["/bin/bash", "-c", "start_local_redis() {" + function
                 + "\n}\nstart_local_redis"],
                env={"REDIS_URL": url, "PATH": "/nonexistent"},
                capture_output=True, text=True,
            )
            with self.subTest(url=url):
                self.assertEqual(result.returncode, 1)
                self.assertIn("External Redis is disabled", result.stderr)
                self.assertNotIn(url, result.stdout + result.stderr)

    def test_local_redis_startup_keeps_bind_and_storage_protections(self):
        source = (Path(__file__).resolve().parents[1] / "src/start.sh").read_text()
        function = source.split("start_local_redis() {", 1)[1].split("\n}\n", 1)[0]
        stubs = '''
attempt=0
redis-cli() {
    [[ "$*" == '-u redis://127.0.0.1:6379 ping' ]] || exit 2
    if [ "$attempt" = 0 ]; then return 1; fi
    echo PONG
}
redis-server() {
    [[ "$*" == '--daemonize yes --bind 127.0.0.1 --protected-mode yes --save  --appendonly no' ]] || exit 2
}
sleep() { attempt=1; }
'''
        result = subprocess.run(
            ["/bin/bash", "-c", stubs + "start_local_redis() {" + function
             + "\n}\nstart_local_redis"],
            env={"REDIS_URL": "redis://localhost:6379/0", "PATH": "/nonexistent"},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Redis ready at redis://127.0.0.1:6379", result.stdout)


if __name__ == "__main__":
    unittest.main()
