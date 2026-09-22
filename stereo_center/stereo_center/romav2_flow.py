"""RoMa v2 dense normalized correspondences to pixel temporal flow."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_to_flow(warp: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if warp.ndim != 4 or warp.shape[-1] != 2:
        raise ValueError("warp must be B,H,W,2")
    h, w = warp.shape[1:3]
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=warp.device, dtype=torch.float32)+.5)*2/h-1,
        (torch.arange(w, device=warp.device, dtype=torch.float32)+.5)*2/w-1,
        indexing="ij")
    identity = torch.stack((xx, yy), -1)[None]
    delta = (warp.float()-identity).permute(0, 3, 1, 2)
    delta = F.interpolate(delta, size=(height, width), mode="bilinear", align_corners=False)
    return delta * delta.new_tensor([width/2, height/2])[None, :, None, None]


class RoMaFlow:
    def __init__(self, setting="fast"):
        from romav2 import RoMaV2
        torch.set_float32_matmul_precision("highest")
        self.model = RoMaV2()
        self.model.apply_setting(setting)
        self.model.bidirectional = True
        self.setting = setting

    @torch.inference_mode()
    def pair(self, previous, current):
        if previous.shape != current.shape or previous.ndim != 4 or previous.shape[1] != 3:
            raise ValueError("RGB images must have equal B,3,H,W shape")
        for image in (previous, current):
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 255:
                raise ValueError("Input must contain finite RGB in [0,255]")
        pred = self.model.match(previous.float()/255, current.float()/255)
        h, w = previous.shape[-2:]
        flows = [warp_to_flow(pred[f"warp_{direction}"], h, w) for direction in ("AB", "BA")]
        overlap = [F.interpolate(pred[f"overlap_{direction}"].permute(0, 3, 1, 2),
                                 size=(h, w), mode="bilinear", align_corners=False)
                   for direction in ("AB", "BA")]
        if any(not torch.isfinite(value).all() for value in flows + overlap):
            raise RuntimeError("RoMa returned nonfinite values")
        return (*flows, *overlap)
