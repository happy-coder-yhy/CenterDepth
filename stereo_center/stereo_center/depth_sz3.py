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


class DepthLogSZ3(Codec):
    """S scheme: metric float32 API, near-linear/far-log SZ3, exact exceptions.

    For 0 < z <= 10 m, |decoded-z| <= alpha * max(1, z) m.
    Other float32 values are stored bitwise losslessly in the chunk envelope.
    """

    codec_id = "centerdepth.log_sz3.v1"
    _magic = b"CDLOGSZ1"
    _header = struct.Struct("<8sIIII")

    def __init__(self, shape, alpha=0.002):
        self.shape = tuple(shape)
        if not self.shape or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
            for n in self.shape
        ):
            raise ValueError("Chunk shape must have positive integer dimensions")
        self.shape = tuple(int(n) for n in self.shape)
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0.0001 <= self.alpha <= 0.1:
            raise ValueError("alpha must be finite and between 0.0001 and 0.1")
        # Reserve margin for float32 transforms and inverse rounding.
        self.transform_tolerance = (
            0.00199 if self.alpha == 0.002 else 0.995 * np.log1p(self.alpha)
        )
        self._inner = DepthSZ3(self.shape, self.transform_tolerance)

    def encode(self, buf):
        data = np.frombuffer(buf, dtype="<f4").reshape(self.shape)
        normal = np.isfinite(data) & (data > 0) & (data <= 10)
        transformed = np.ones(self.shape, dtype="<f4")
        values = data[normal].astype(np.float64)
        far = values > 1
        values[far] = 1 + np.log(values[far])
        transformed[normal] = values
        payload = self._inner.encode(transformed)
        mask = exceptions = b""
        if not normal.all():
            mask = zlib.compress(np.packbits(~normal.ravel()).tobytes())
            exceptions = zlib.compress(data[~normal].tobytes())
        body = payload + mask + exceptions
        return self._header.pack(
            self._magic, len(payload), len(mask), len(exceptions), zlib.crc32(body)
        ) + body

    def decode(self, buf, out=None):
        encoded = memoryview(buf).cast("B")
        if len(encoded) < self._header.size:
            raise ValueError("Truncated S depth chunk")
        magic, size, mask_size, values_size, checksum = self._header.unpack(
            encoded[:self._header.size]
        )
        body = encoded[self._header.size:]
        if (magic != self._magic or size == 0
                or len(body) != size + mask_size + values_size
                or zlib.crc32(body) != checksum
                or bool(mask_size) != bool(values_size)):
            raise ValueError("Invalid or corrupted S depth chunk")
        exceptional = None
        if mask_size:
            mask = zlib.decompress(body[size:size + mask_size])
            count = int(np.prod(self.shape))
            if len(mask) != (count + 7) // 8:
                raise ValueError("Invalid S depth exception mask")
            exceptional = np.unpackbits(np.frombuffer(mask, dtype=np.uint8))
            exceptional = exceptional[:count].reshape(self.shape).astype(bool)
            original = zlib.decompress(body[size + mask_size:])
            if len(original) != int(exceptional.sum()) * 4:
                raise ValueError("Invalid S depth exception values")
        data = self._inner.decode(body[:size]).astype(np.float64)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite transformed S depth")
        # Projection onto the valid domain cannot increase reconstruction error.
        np.clip(data, 0, 1 + np.log(10), out=data)
        far = data > 1
        data[far] = np.exp(data[far] - 1)
        data = np.clip(data, np.nextafter(np.float32(0), np.float32(1)), 10).astype("<f4")
        if exceptional is not None:
            data.view("<u4")[exceptional] = np.frombuffer(original, dtype="<u4")
        if out is None:
            return data
        target = np.frombuffer(out, dtype="<f4").reshape(self.shape)
        np.copyto(target, data)
        return out

    def get_config(self):
        return {"id": self.codec_id, "shape": list(self.shape), "alpha": self.alpha}


register_codec(DepthLogSZ3)
