"""SZ3 depth codec; importing this module registers it with Numcodecs."""

from __future__ import annotations

import numpy as np
import struct
import zlib
from numcodecs.abc import Codec
from numcodecs.registry import register_codec


class DepthSZ3(Codec):
    """Compress nonnegative finite float32 depth with an absolute error tolerance.

    Readers need pysz and must import this module before opening the Zarr array.
    Zero-valued invalid pixels are preserved with a lossless mask.
    """

    codec_id = "centerdepth.sz3.v1"
    _raw_prefix = b"CDSZ3RAW\x00"
    _masked_prefix = b"CDSZ3MASK\x00"

    def __init__(self, shape, tolerance=0.001):
        self.shape = tuple(int(n) for n in shape)
        self.tolerance = float(tolerance)
        if not self.shape or any(n < 1 for n in self.shape):
            raise ValueError("Chunk shape must have positive dimensions")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Tolerance must be finite and positive")

    def encode(self, buf):
        from pysz import sz, szConfig, szErrorBoundMode

        data = np.frombuffer(buf, dtype="<f4").reshape(self.shape)
        if not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("SZ3 depth codec requires finite nonnegative values")
        config = szConfig()
        config.errorBoundMode = szErrorBoundMode.ABS
        config.absErrorBound = self.tolerance
        try:
            compressed, _ = sz.compress(np.ascontiguousarray(data), config)
            payload = compressed.tobytes()
        except ValueError as exc:
            # PySZ 1.0.3 may undersize the output buffer for small chunks.
            if "buffer for compressed data is not large enough" not in str(exc):
                raise
            payload = self._raw_prefix + data.tobytes()
        invalid = data == 0
        if invalid.any():
            mask = zlib.compress(np.packbits(invalid.ravel()).tobytes())
            return self._masked_prefix + struct.pack("<I", len(mask)) + mask + payload
        return payload

    def decode(self, buf, out=None):
        from pysz import sz

        encoded = memoryview(buf).cast("B")
        invalid = None
        if bytes(encoded[:len(self._masked_prefix)]) == self._masked_prefix:
            offset = len(self._masked_prefix)
            mask_size = struct.unpack("<I", encoded[offset:offset + 4])[0]
            offset += 4
            mask = zlib.decompress(encoded[offset:offset + mask_size])
            invalid = np.unpackbits(np.frombuffer(mask, dtype=np.uint8))
            invalid = invalid[:int(np.prod(self.shape))].reshape(self.shape).astype(bool)
            encoded = encoded[offset + mask_size:]
        if bytes(encoded[:len(self._raw_prefix)]) == self._raw_prefix:
            data = np.frombuffer(encoded[len(self._raw_prefix):], dtype="<f4").reshape(self.shape)
        else:
            compressed = np.frombuffer(encoded, dtype=np.uint8).copy()
            data, _ = sz.decompress(compressed, np.float32, self.shape)
        data = np.array(data, dtype="<f4", order="C", copy=True)
        np.maximum(data, np.finfo(np.float32).tiny, out=data)
        if invalid is not None:
            data[invalid] = 0.0
        if out is None:
            return data
        target = np.frombuffer(out, dtype="<f4").reshape(self.shape)
        np.copyto(target, data)
        return out

    def get_config(self):
        return {"id": self.codec_id, "shape": list(self.shape), "tolerance": self.tolerance}


register_codec(DepthSZ3)
