import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import comfy_client


class FakeResponse:
    def __init__(self, status=200, json_body=None):
        self.status = status
        self._json_body = json_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self._json_body


class TestCheckServer(unittest.TestCase):
    @patch("comfy_client.asyncio.sleep", new_callable=AsyncMock)
    def test_retries_after_connection_error_then_succeeds(self, mock_sleep):
        session = Mock()
        session.get = Mock(
            side_effect=[ConnectionError("refused"), FakeResponse(status=200)]
        )

        asyncio.run(comfy_client.check_server(session, "127.0.0.1:8188", timeout_s=10, interval_s=1))

        self.assertEqual(session.get.call_count, 2)
        mock_sleep.assert_awaited()

    @patch("comfy_client.asyncio.sleep", new_callable=AsyncMock)
    def test_timeout_raises_and_never_submits_prompt(self, mock_sleep):
        session = Mock()
        session.get = Mock(side_effect=ConnectionError("refused"))
        session.post = Mock()

        with self.assertRaises(TimeoutError):
            asyncio.run(
                comfy_client.run_graph(session, "127.0.0.1:8188", {}, start_time=0, ready_timeout_s=2)
            )

        session.post.assert_not_called()


class TestQueuePrompt(unittest.TestCase):
    def test_node_errors_are_named_in_the_exception(self):
        session = Mock()
        session.post = Mock(
            return_value=FakeResponse(
                status=200,
                json_body={
                    "error": {"type": "invalid_prompt"},
                    "node_errors": {"398:376": {"errors": ["required input missing"]}},
                },
            )
        )

        with self.assertRaisesRegex(RuntimeError, "398:376"):
            asyncio.run(comfy_client.queue_prompt(session, "127.0.0.1:8188", {}))

        session.post.assert_called_once()


class TestPollHistory(unittest.TestCase):
    @patch("comfy_client.asyncio.sleep", new_callable=AsyncMock)
    @patch("comfy_client.time.time")
    def test_deadline_exceeded_raises_timeout_error(self, mock_time, mock_sleep):
        mock_time.side_effect = [0, 10, 20, 1000]
        session = Mock()
        session.get = Mock(return_value=FakeResponse(status=200, json_body={}))

        with self.assertRaises(TimeoutError):
            asyncio.run(
                comfy_client.poll_history(
                    session, "127.0.0.1:8188", "prompt-1", start_time=0, timeout_s=900
                )
            )


if __name__ == "__main__":
    unittest.main()
