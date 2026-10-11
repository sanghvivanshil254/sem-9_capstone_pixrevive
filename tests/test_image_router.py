import io
import time
import os
import unittest
from fastapi.testclient import TestClient
from main import app
from routers import image_router


class TestImageRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("models", data)
        self.assertIn("ready", data)

    def test_presets_endpoint(self):
        """Q-8: presets are served with labels and the pipeline toggles."""
        response = self.client.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        presets = response.json().get("presets", {})
        self.assertIn("default", presets)
        for preset in presets.values():
            self.assertIn("label", preset)
            self.assertIn("toggles", preset)
            for toggle in (
                "enable_repair",
                "enable_denoise",
                "enable_colorization",
                "enable_face_restore",
                "enable_sr",
            ):
                self.assertIn(toggle, preset["toggles"])
        # The default preset must be the full pipeline.
        self.assertTrue(all(presets["default"]["toggles"].values()))
        # UI-simplification contract (2026-10-01): the default preset is
        # presented as "Auto (recommended)" — the pipeline router already
        # adapts per image, so a separate "Auto" preset would be a duplicate.
        # Single-stage presets (repair/colorize/denoise only) were removed
        # from the selector; per-stage control lives in More Options.
        self.assertEqual(presets["default"]["label"], "Auto (recommended)")
        self.assertNotIn("repair_only", presets)
        self.assertNotIn("colorize_only", presets)
        self.assertNotIn("denoise_only", presets)
        self.assertIn("fast", presets)
        self.assertIn("reimagine_beta", presets)

    def test_upload_valid_png(self):
        # 1x1 transparent PNG bytes
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.png", io.BytesIO(png_bytes), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("filename"), "test.png")
        self.assertEqual(data.get("content_type"), "image/png")

    def test_upload_invalid_mime_type(self):
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json().get("detail", ""))

    def test_upload_file_too_large(self):
        # Construct virtual file exceeding 25 MB limit
        large_bytes = b"0" * (25 * 1024 * 1024 + 1024)
        response = self.client.post(
            "/api/upload",
            files={"file": ("large.png", io.BytesIO(large_bytes), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("File too large", response.json().get("detail", ""))

    def test_status_nonexistent_task(self):
        response = self.client.get("/api/process/fake_task_id_123/status")
        self.assertEqual(response.status_code, 404)

    def test_task_pruning_lifecycle(self):
        # Insert artificial expired task and assert eviction
        tid = "test_expired_tid"
        test_file = os.path.join(image_router._results_dir, f"{tid}.png")
        with open(test_file, "wb") as f:
            f.write(b"dummy png content")

        with image_router._tasks_lock:
            image_router._tasks[tid] = {
                "status": "done",
                "result_path": test_file,
                "created_at": time.time() - 2000,  # Expired (>1800s)
            }

        image_router._prune_old_tasks()

        with image_router._tasks_lock:
            self.assertNotIn(tid, image_router._tasks)
        self.assertFalse(os.path.exists(test_file))

    def test_get_status_prunes_expired_tasks(self):
        """U-9: every status read is a pruning opportunity — expired tasks are
        evicted before the lookup and their result files deleted."""
        tid = "test_status_prune_tid"
        test_file = os.path.join(image_router._results_dir, f"{tid}.png")
        with open(test_file, "wb") as f:
            f.write(b"dummy png content")

        with image_router._tasks_lock:
            image_router._tasks[tid] = {
                "status": "done",
                "result_path": test_file,
                "created_at": time.time() - 2000,  # Expired (>1800s)
            }

        # The read prunes first: expired task answers 404 and is gone.
        response = self.client.get(f"/api/process/{tid}/status")
        self.assertEqual(response.status_code, 404)

        with image_router._tasks_lock:
            self.assertNotIn(tid, image_router._tasks)
        self.assertFalse(os.path.exists(test_file))
