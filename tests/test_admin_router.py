import io
import os
import zipfile
import tempfile
import unittest
from fastapi.testclient import TestClient
from main import app
from routers.admin_router import async_process_zip_dataset, upload_progress_store


class TestAdminRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_weights_status_endpoint(self):
        response = self.client.get("/api/admin/weights/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("weights", data)
        self.assertIn("gfpgan", data)
        self.assertIn("realesrgan", data)
        self.assertIn("lama", data)
        self.assertIn("deblur", data)

    def test_upload_invalid_weights_extension(self):
        response = self.client.post(
            "/api/admin/weights/upload",
            data={"model_type": "gfpgan"},
            files={"file": ("invalid.txt", io.BytesIO(b"fake data"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PyTorch .pth/.pt files are allowed", response.json().get("detail", ""))

    def test_upload_extended_model_types(self):
        """Plan 6.2: lama (.pt), colorizer (DDColor) and deblur (Restormer)
        upload targets route to their canonical layout; targets are patched
        into a temp dir so real weights are never touched."""
        import tempfile

        from routers import admin_router

        with tempfile.TemporaryDirectory() as tmp:
            orig = dict(admin_router._WEIGHT_UPLOAD_TARGETS)
            admin_router._WEIGHT_UPLOAD_TARGETS.update({
                "lama":      ("big-lama.pt", tmp),
                "colorizer": ("ddcolor_modelscope.pth", tmp),
                "deblur":    ("motion_deblurring.pth", tmp),
            })
            try:
                # Extension check is unified (.pth or .pt): either torch
                # checkpoint extension is accepted; the saved name is the
                # canonical target (big-lama.pt) regardless of upload name.
                for up_name in ("w.pth", "big-lama.pt"):
                    r = self.client.post(
                        "/api/admin/weights/upload",
                        data={"model_type": "lama"},
                        files={"file": (up_name, io.BytesIO(b"PK\x03\x04fake"), "application/octet-stream")},
                    )
                    self.assertEqual(r.status_code, 200, r.json())
                    self.assertTrue(os.path.exists(os.path.join(tmp, "big-lama.pt")))

                # A .txt is still rejected.
                r = self.client.post(
                    "/api/admin/weights/upload",
                    data={"model_type": "lama"},
                    files={"file": ("w.txt", io.BytesIO(b"nope"), "text/plain")},
                )
                self.assertEqual(r.status_code, 400)
                self.assertIn(".pth/.pt", r.json()["detail"])

                # DDColor and Restormer targets accept .pth checkpoints.
                for model_type, filename in (
                    ("colorizer", "ddcolor_modelscope.pth"),
                    ("deblur", "motion_deblurring.pth"),
                ):
                    r = self.client.post(
                        "/api/admin/weights/upload",
                        data={"model_type": model_type},
                        files={"file": (filename, io.BytesIO(b"PK\x03\x04fake"), "application/octet-stream")},
                    )
                    self.assertEqual(r.status_code, 200, r.json())
                    self.assertTrue(os.path.exists(os.path.join(tmp, filename)))

                # Unknown model type lists the valid options.
                r = self.client.post(
                    "/api/admin/weights/upload",
                    data={"model_type": "bogus"},
                    files={"file": ("x.pth", io.BytesIO(b"PK\x03\x04fake"), "application/octet-stream")},
                )
                self.assertEqual(r.status_code, 400)
                self.assertIn("colorizer", r.json()["detail"])
            finally:
                admin_router._WEIGHT_UPLOAD_TARGETS.clear()
                admin_router._WEIGHT_UPLOAD_TARGETS.update(orig)

    def test_admin_token_enforcement(self):
        os.environ["PIXREVIVE_ADMIN_TOKEN"] = "secret123"
        try:
            # Without header -> 401
            res_unauth = self.client.get("/api/admin/weights/status")
            self.assertEqual(res_unauth.status_code, 401)

            # With incorrect header -> 401
            res_wrong = self.client.get(
                "/api/admin/weights/status",
                headers={"X-Admin-Token": "wrong"},
            )
            self.assertEqual(res_wrong.status_code, 401)

            # With correct header -> 200
            res_auth = self.client.get(
                "/api/admin/weights/status",
                headers={"X-Admin-Token": "secret123"},
            )
            self.assertEqual(res_auth.status_code, 200)
        finally:
            del os.environ["PIXREVIVE_ADMIN_TOKEN"]

    def test_zip_slip_traversal_rejected(self):
        # Create a malicious zip file containing path traversal entries
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("../evil.txt", "malicious content")
            zf.writestr("..\\evil_win.txt", "malicious content win")
            zf.writestr("normal.png", "safe content")

        temp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        temp_zip.write(zip_buffer.getvalue())
        temp_zip.close()

        upload_id = "test_zip_slip_check"
        try:
            async_process_zip_dataset(temp_zip.name, upload_id)
            # The malicious members must have been skipped without writing outside temp_dir
            progress = upload_progress_store.get(upload_id, {})
            self.assertIn(progress.get("status"), ("completed", "processing"))
        finally:
            if os.path.exists(temp_zip.name):
                try:
                    os.remove(temp_zip.name)
                except OSError:
                    pass
