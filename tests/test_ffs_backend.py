from pathlib import Path
import tempfile
import unittest

from stereo_center import stereo_backend
from stereo_center import ffs_inference


class FFSBackendTests(unittest.TestCase):
    def test_backend_registry_includes_ffs(self):
        self.assertIn("ffs", stereo_backend.BACKENDS)
        self.assertIs(stereo_backend.get_backend("ffs"), ffs_inference)

    def test_resolve_checkpoint_accepts_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "model_best_bp2_serialize.pth"
            ckpt.write_bytes(b"placeholder")

            self.assertEqual(ffs_inference.resolve_checkpoint(root), ckpt)

    def test_resolve_checkpoint_accepts_direct_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "custom_model.pth"
            ckpt.write_bytes(b"placeholder")

            self.assertEqual(ffs_inference.resolve_checkpoint(ckpt), ckpt)


if __name__ == "__main__":
    unittest.main()
