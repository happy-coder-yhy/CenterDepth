# StereoNet_PyTorch Upstream Provenance

## Pinned source

- Repository: <https://github.com/andrewlstewart/StereoNet_PyTorch>
- Pinned commit: `9c0260f270547d8001e9d637cf3a94658f805bae`
- License: The Unlicense (the upstream `LICENSE` file)

This repository is a third-party reproduction of StereoNet, not an official
release by the StereoNet paper authors.  Treat its implementation, training
configuration, and checkpoint as third-party artifacts.

## Reported checkpoint configuration

- Checkpoint URL:
  <https://www.dropbox.com/s/9gpjfe3r1rfch02/epoch%3D20-step%3D744533.ckpt?dl=0>
- Verified checkpoint SHA-256:
  `03b67d8571f39505959cf485de272fe0ea615a1d8dd3fab16f06af4acec2b82e`
- Upstream-reported maximum disparity: `256` (with the training mask applied)
- Upstream-reported validation EPE: `3.93` for all pixels, including values
  greater than 256

The optional local checkout lives at
`stereo_center/third_party/StereoNet_PyTorch` and must remain uncommitted.
Checkpoint-specific provenance, including its SHA-256 and serialized key
layout, is recorded in `weights/stereonet/README.md` after the checkpoint has
been obtained and inspected.
