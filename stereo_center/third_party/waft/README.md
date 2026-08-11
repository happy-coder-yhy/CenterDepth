# WAFT-Stereo (vendored, inference-only subset)

Vendored from https://github.com/princeton-vl/WAFT-Stereo
(commit `c534320860f4bbb2c0df896e3ccb95de49284aec`, 2026-05-05, main branch)
for local inference. License: BSD 3-Clause (see `LICENSE`).

Included minimal subset needed at inference time:

- `algorithms/waft.py` — WAFT model (forward / inference / hierarchical inference)
- `model/` — DAv2/DINOv3 encoders and ViT iterative modules
- `bridgedepth/config/` — yacs-based config loading
- `thirdparty/DepthAnythingV2/` — Depth-Anything-V2 DPT/DINOv2 code (random init
  at construction; weights come from the WAFT-Stereo checkpoint)
- `configs/SynLarge/DAv2{S,B,L}-*.yaml` — SynLarge configs for the released
  zero-shot checkpoints

Only local patch vs upstream: `model/iterative/vit.py` uses `pretrained=False`
for the timm ViT to avoid a network download at model construction.
