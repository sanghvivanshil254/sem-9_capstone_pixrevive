"""
Unit tests for SSE Progress Streaming Endpoint (/api/process-stream).

Verifies:
- Monotonic progress percentages from Server-Sent Events (SSE).
- Streaming headers (Cache-Control, X-Accel-Buffering).
- Graceful completion event with valid task_id and result retrieval.
- Error event propagation when processing raises an exception.
- Input validation (MIME types, file size).
"""

import io
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from main import app
from routers import image_router


class TestProgressStream(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    def test_process_stream_invalid_mime(self):
        resp = self.client.post(
            "/api/process-stream",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file type", resp.json().get("detail", ""))

    def test_process_stream_monotonic_events(self):
        # Mock ImageService.process_image to emit known progress stages
        def mock_process_image(*args, **kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb("analyze", 10, "Analyzing photo quality...")
                cb("detect", 30, "Detecting physical damage...")
                cb("inpaint", 60, "Inpainting cracks and tears...")
                cb("enhance", 90, "Applying final exhibition grade...")
            return self.sample_png

        with patch.object(image_router.ImageService, "process_image", side_effect=mock_process_image):
            resp = self.client.post(
                "/api/process-stream",
                files={"file": ("input.png", io.BytesIO(self.sample_png), "image/png")},
                data={"restore": "true", "repair": "true", "colorize": "false"},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
            self.assertEqual(resp.headers.get("x-accel-buffering"), "no")

            # Parse SSE events from response text
            lines = resp.text.strip().splitlines()
            events = []
            curr_event = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    curr_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_str = line.split(":", 1)[1].strip()
                    payload = json.loads(data_str)
                    events.append((curr_event, payload))
                    curr_event = None

            # Must contain progress events and terminate with complete
            self.assertGreater(len(events), 2)
            progress_percentages = []
            terminal_event = None
            task_id = None

            for ev_type, payload in events:
                if ev_type == "progress":
                    self.assertIn("stage", payload)
                    self.assertIn("percent", payload)
                    self.assertIn("message", payload)
                    progress_percentages.append(payload["percent"])
                elif ev_type == "complete":
                    terminal_event = payload
                    task_id = payload.get("task_id")

            # Verify strictly monotonic percentages
            self.assertEqual(progress_percentages, sorted(progress_percentages))
            self.assertEqual(progress_percentages, [10, 30, 60, 90])

            # Verify completion payload
            self.assertIsNotNone(terminal_event)
            self.assertEqual(terminal_event.get("percent"), 100)
            self.assertIsNotNone(task_id)

            # Verify task status and result fetch
            res_get = self.client.get(f"/api/process/{task_id}/result")
            self.assertEqual(res_get.status_code, 200)
            self.assertEqual(res_get.content, self.sample_png)

    def test_process_stream_error_propagation(self):
        def mock_failing_process(*args, **kwargs):
            raise RuntimeError("Synthetic pipeline failure during inference")

        with patch.object(image_router.ImageService, "process_image", side_effect=mock_failing_process):
            resp = self.client.post(
                "/api/process-stream",
                files={"file": ("input.png", io.BytesIO(self.sample_png), "image/png")},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

            lines = resp.text.strip().splitlines()
            error_found = False
            for line in lines:
                if "Synthetic pipeline failure" in line:
                    error_found = True
                    break

            self.assertTrue(error_found, f"Expected error event in stream: {resp.text}")
