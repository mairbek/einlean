#!/usr/bin/env python3
import ast
import struct
from pathlib import Path


def read_npy_f64(path: Path):
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("Not an NPY file")
        major, minor = struct.unpack("BB", f.read(2))
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            hlen = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"Unsupported NPY version {major}.{minor}")
        header = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        if header.get("descr") != "<f8":
            raise ValueError(f"Expected <f8, got {header.get('descr')}")
        if header.get("fortran_order"):
            raise ValueError("Fortran-order arrays are not supported")
        shape = tuple(header.get("shape"))
        total = 1
        for s in shape:
            total *= s
        vals = struct.unpack("<" + "d" * total, f.read(total * 8))
        return shape, vals


def to_byte(x: float) -> int:
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 255
    return int(round(x * 255.0))


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "test_images.npy"
    dst = root / "test_images.rgb"

    shape, vals = read_npy_f64(src)
    if shape != (6, 96, 96, 3):
        raise ValueError(f"Expected shape (6,96,96,3), got {shape}")

    b, h, w, c = shape

    def src_idx(bi: int, hi: int, wi: int, ci: int) -> int:
        return ((bi * h + hi) * w + wi) * c + ci

    out = bytearray()
    for bi in range(b):
        for wi in range(w):
            for hi in range(h):
                for ci in range(c):
                    out.append(to_byte(vals[src_idx(bi, hi, wi, ci)]))

    dst.write_bytes(out)
    print(f"Wrote {dst} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
