# Future Ideas

## Compile-time size checking for tensor literals

Use a custom macro like `t![[1,2,3],[4,5,6]]` that parses nested list literals, checks that dimensions match the expected tensor shape at elaboration time, and produces a `Tensor.ofData` call. Gives both nice syntax and compile-time safety.

## Dim-first tensor design (rearrange v2)

### Problem

Current rearrange uses Slots (type-level position markers) inside a lambda. This prevents using named dim expressions like `bw := b * w` inside the lambda body, because `bw` is a `Dim` while the lambda params are `Slot`s.

### Core idea: Dim as ordered atom list

A `Dim` is a flat, ordered list of atoms. Each atom has an `id` (identity) and a `size`.

```lean
structure Atom where
  id : Nat
  size : Nat

structure Dim where
  atoms : List Atom

def dim (id : Nat) (size : Nat) : Dim := ⟨[⟨id, size⟩]⟩

instance : Mul Dim where
  mul d1 d2 := ⟨d1.atoms ++ d2.atoms⟩
```

Key properties of this model:

- **`b * w ≠ w * b`**: different atom order → different data layout (last atom varies fastest)
- **`(b * w) * c = b * (w * c)`**: `List.append` is associative → same normal form, same Lean type
- **`(Dim, *)` is a free monoid**: non-commutative, associative

### No lambda needed

Dims carry identity, so the system matches output to input by atom `id`. No Slots, no lambda:

```lean
def b := dim 0 32
def h := dim 1 224
def w := dim 2 224
def c := dim 3 3
def bw := b * w

def image : Tensor [b, h, w, c] := Tensor.zeros
def merged : Tensor [h, bw, c] := image.rearrange
```

Compile-time validation via `by decide`:

```lean
def Tensor.rearrange {outDims : DimList} (t : Tensor inDims)
    (valid : validRearrange inDims outDims = true := by decide)
    : Tensor outDims
```

Invalid rearrangements fail at compile time — `decide` can't prove `false = true`.

### Atom as sampling of a range (future: crop/resize)

To support operations beyond rearrange (crop, resize, pad), the atom model extends to describe a sampling of a coordinate range:

```
Atom { id, size, rangeStart, rangeEnd }
```

| expression      | id | size | range     | meaning                              |
|-----------------|----|------|-----------|--------------------------------------|
| `h`             | 1  | 224  | [0, 224)  | full axis                            |
| `h[0:128]`      | 1  | 128  | [0, 128)  | crop — range shrinks with size       |
| `h[96:224]`     | 1  | 128  | [96, 224) | crop — different region              |
| `h.resize(128)` | 1  | 128  | [0, 224)  | resize — range unchanged, fewer samples |

The ratio `size / (rangeEnd - rangeStart)` is the sampling density:
- Density = 1 → crop (pure index selection)
- Density ≠ 1 → interpolation needed (resize)

Different operations use different matching:
- **Rearrange**: exact atom match (id + size + range) — pure reordering
- **Crop/Resize**: id match only — operation handles the transformation

### Einsum (same approach)

No lambda needed. Contracted dims are those appearing in both inputs but not the output:

```lean
def cmat : Tensor [i, j] := Tensor.einsum a bmat
-- a : Tensor [i, k], bmat : Tensor [k, j]
-- k in both inputs but not output → contracted
```
