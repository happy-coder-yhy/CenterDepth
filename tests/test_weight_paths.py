import tempfile
import unittest
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "stereo_center"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from stereo_center.weight_paths import resolve_depth_anything_checkpoint


class DepthAnythingWeightPathTests(unittest.TestCase):
    def test_repository_weights_win_over_legacy_stereo_center_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            module_root = repo_root / "stereo_center"
            canonical = repo_root / "weights" / "depth-anything-ckpts"
            legacy = module_root / "weights" / "depth-anything-ckpts"
            canonical.mkdir(parents=True)
            legacy.mkdir(parents=True)
            canonical_checkpoint = canonical / "depth_anything_v2_vits.pth"
            legacy_checkpoint = legacy / "depth_anything_v2_vits.pth"
            canonical_checkpoint.touch()
            legacy_checkpoint.touch()

            resolved = resolve_depth_anything_checkpoint("vits", module_root)

            self.assertEqual(resolved, canonical_checkpoint)
