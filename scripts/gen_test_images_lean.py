#!/usr/bin/env python3
import ast
import struct
from pathlib import Path


def read_npy_f64(path: Path):
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
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
            raise ValueError(f"Expected <f8 dtype, got {header.get('descr')}")
        if header.get("fortran_order"):
            raise ValueError("Fortran-order arrays are not supported")

        shape = header.get("shape")
        total = 1
        for s in shape:
            total *= s

        data = f.read(total * 8)
        values = struct.unpack("<" + "d" * total, data)
        return shape, values


def f01_to_byte(x: float) -> int:
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 255
    return int(round(x * 255.0))


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "test_images.npy"
    out = root / "Einlean" / "TestImagesData.lean"

    shape, values = read_npy_f64(src)
    if tuple(shape) != (6, 96, 96, 3):
        raise ValueError(f"Unexpected shape {shape}; expected (6, 96, 96, 3)")

    ints = [f01_to_byte(v) for v in values]
    b, h, w, c = shape

    lines = []
    lines.append("import Einlean")
    lines.append("")
    lines.append("namespace Einlean")
    lines.append("")
    lines.append("def testB := dim! 6")
    lines.append("def testW := dim! 96")
    lines.append("def testH := dim! 96")
    lines.append("def testC := dim! 3")
    lines.append("")
    lines.append(
        "/-- Generated from test_images.npy (shape 6x96x96x3, float64 in [0,1]). -/"
    )

    # input is [b, h, w, c], output expected [b, w, h, c]
    # remap by swapping h and w
    # create a direct indexer into flat [b,h,w,c]
    def src_index(bi: int, hi: int, wi: int, ci: int) -> int:
        return ((bi * h + hi) * w + wi) * c + ci

    reordered = []
    for bi in range(b):
        for wi in range(w):
            for hi in range(h):
                for ci in range(c):
                    reordered.append(ints[src_index(bi, hi, wi, ci)])

    chunk_size = 1024
    chunk_names = []
    for i in range(0, len(reordered), chunk_size):
        name = f"testChunk{i // chunk_size}"
        chunk_names.append(name)
        chunk = reordered[i : i + chunk_size]
        lines.append(f"private def {name} : List Int := [")
        for j, v in enumerate(chunk):
            sep = "," if j + 1 < len(chunk) else ""
            lines.append(f"  {v}{sep}")
        lines.append("]")
        lines.append("")

    lines.append("private def testImageData : List Int :=")
    lines.append("  List.join [")
    for i, name in enumerate(chunk_names):
        sep = "," if i + 1 < len(chunk_names) else ""
        lines.append(f"    {name}{sep}")
    lines.append("  ]")
    lines.append("")
    lines.append("def testImages : Tensor [testB, testW, testH, testC] Int :=")
    lines.append("  Tensor.ofData (dims := [testB, testW, testH, testC]) testImageData")
    lines.append("")
    lines.append("end Einlean")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
