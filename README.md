# Einlean Design

Compile-time verifiable tensor operations in Lean 4, einops-style.

This document describes the current implementation in `Einlean.lean`.

## High-Level Architecture

Einlean has two layers:

1. **Type-level dimension algebra** (`Atom`, `Dim`, `DimList`) used to prove shape compatibility.
2. **Runtime tensor layout** (`Tensor`) that stores contiguous data plus shape/stride metadata.

Core transforms (`rearrange`, `reshape`, `einsum`) use compile-time constraints, then execute explicit index-mapping loops at runtime.

## Dimension Model

### Atom and Dim

```lean
structure Atom where
  id : Nat
  size : Nat
  rank : Nat := 0

structure Dim where
  atoms : List Atom
```

- A `Dim` is a product of one or more `Atom`s.
- `id` is identity, `size` is extent, `rank` is used to order factored atoms.

### Constructors and Composition

```lean
def dim (id : Nat) (size : Nat) : Dim :=
  ⟨[{ id := id, size := size, rank := 0 }]⟩

scoped macro "dim!" size:term : term => ...

instance : Mul Dim where
  mul d1 d2 := ⟨normalizeAtoms (d1.atoms ++ d2.atoms)⟩
```

- `dim!` creates source-position-unique IDs at elaboration time.
- `d1 * d2` concatenates atom lists, then normalizes adjacent equal IDs.
- `Dim.size` multiplies atom sizes.

### Factoring

```lean
def Dim.factor! (d : Dim) (k : Nat) ... : Dim
```

- `factor!` splits a 1-atom dim into ranked atoms to enable reversible batch-style regrouping.

## Tensor Runtime Model

```lean
structure Tensor (dims : DimList) (α : Type := Int) where
  data : Array α
  shape : Array Nat
  strides : Array Nat
  offset : Nat := 0
```

- `dims` is the compile-time shape descriptor.
- `shape`/`strides`/`offset` drive runtime indexing.
- Backing storage is a flat `Array α`.

Key helpers:

- `computeStrides`, `flatIndex`, `toMultiIndex`, `shapeOf`, `totalSize`
- constructors: `ofFn`, `ofArray`, `ofData`, `fill`, `zeros`, `arange`
- views/access: `get!`, `slice0`, `GetElem` instance, `toList`

## Rearrange: Two APIs

### 1) Lambda-based (`rearrangeBy`)

```lean
def Tensor.rearrangeBy {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    (t : Tensor dims α) (_f : SlotTuple dims → Out)
    : Tensor h.outDims α
```

Supporting pieces:

- `Slot dims i`: zero-sized positional marker.
- `MergedSlot dims is`: marker for merged output axes.
- `HasDims Out dims` computes:
  - `outDims : DimList`
  - `outMapping : List (List Nat)` (which input axes feed each output axis)

Execution path:

1. `HasDims` derives `outMapping` from the lambda return type.
2. `Tensor.rearrangeCore` builds output shape and remaps indices using that mapping.

This path is expressive and supports syntax like:

```lean
image.rearrangeBy fun (b, h, w, c) => (h, b * w, c)
```

### 2) Lambda-free (`rearrange`)

```lean
def Tensor.rearrange {inDims outDims : DimList} (t : Tensor inDims α)
    (_valid : validRearrange inDims outDims = true := by decide)
    : Tensor outDims α
```

Compile-time guard:

```lean
def validRearrange (inDims outDims : DimList) : Bool
```

It checks:

- same set of atom IDs
- same per-ID size products

Runtime path (`rearrangeByRoots`):

1. Flatten input/output dims to atom streams.
2. Group atoms by root `id` into `RootSpec`.
3. Convert each output index -> output atom digits.
4. Transport coordinates per root into input atom digits.
5. Reassemble input axis indices and read source element.

This enables rearrangements/merges/splits without an explicit lambda when the output type is known.

## Reshape

```lean
def Tensor.reshape {inDims outDims : DimList} (t : Tensor inDims α)
    (_valid : validReshape inDims outDims = true := by decide)
    : Tensor outDims α
```

`validReshape` requires exact atom-stream equality (`allAtoms inDims == allAtoms outDims`).

- This is stricter than `rearrange`.
- Data is not copied; only metadata (`shape`, `strides`) is rebuilt.

## Einsum (Current)

Einlean now exposes two complementary einsum APIs:

### 1) Typed tuple API (`einsum`)

```lean
def Tensor.einsum {Ops : Type} {outDims : DimList}
    [h : EinsumArgs Ops outDims α]
    (ops : Ops)
    (_valid : h.validDims = true := by decide)
    : Tensor outDims α
```

- Supports 2 or 3 operands via tuples:
  - `(x, y)`
  - `(x, y, z)`
- Output dims are usually inferred from the expected type.
- Dimension compatibility is checked at compile time via `validEinsum2` / `validEinsum3`.

### 2) Lambda-based API (`einsumBy`)

```lean
def Tensor.einsumBy {A B : DimList} {Out : Type}
    [h : EinsumOut Out A B]
    (x : Tensor A α) (y : Tensor B α)
    (_f : SlotTuple A → SlotTuple B → Out)
    : Tensor h.outDims α
```

- Binary-only, slot-based output specification.
- Useful when you want explicit output-axis selection in code.

Runtime execution uses a generic contraction kernel that maps output labels vs reduced labels, then accumulates products across contraction coordinates.

## Visualization

`Einlean/Viz.lean` provides:

- `Tensor.toHtmlImage` for `[h, w, c]` tensors
- `Tensor.toHtmlBatch` for `[b, h, w, c]` tensors
- `#imgtensor` command (ProofWidgets-based) for inline rendering in Lean editors

## Usage Examples (Current API)

Note: examples in `Einlean.lean` keep `#eval` on small tensors for build speed; larger tensors are usually kept as typed declarations without runtime evaluation.

```lean
def b := dim! 32
def h := dim! 224
def w := dim! 224
def c := dim! 3
def image : Tensor [b, h, w, c] := Tensor.zeros

-- Lambda-based
def transposed : Tensor [b, c, h, w] :=
  image.rearrangeBy fun (b, h, w, c) => (b, c, h, w)

-- Lambda-free (type drives transform)
def bw := b * w
def merged : Tensor [h, bw, c] := image.rearrange

-- Einsum matmul
def i := dim! 2
def j := dim! 4
def k := dim! 3
def a : Tensor [i, k] := arange 1
def bmat : Tensor [k, j] := arange 10
def cmat : Tensor [i, j] := Tensor.einsum (a, bmat)

-- 3-input einsum (bilinear-style contraction)
def l := dim! 2
def x : Tensor [i, k] := arange 1
def w : Tensor [j, k, l] := arange 1
def y : Tensor [i, l] := arange 1
def bilinear : Tensor [i, j] := Tensor.einsum (x, w, y)

-- Complementary lambda-style API
def cmatBy : Tensor [i, j] :=
  Tensor.einsumBy a bmat (fun (i, _) (_, j) => (i, j))
```

## Current Guarantees and Limits

Guarantees:

- Output tensor types are compile-time checked.
- `rearrange`/`reshape` reject invalid dimension relations via `by decide` proofs.
- Dim composition carries structure through atom IDs/ranks.

Current limits:

- `einsum` currently supports 2 or 3 operands (not arbitrary N in the public API).
- No explicit diagonal/repeated-index semantics yet (relative to full symbolic einsum strings).
- Slot-based APIs (`rearrangeBy` / `einsumBy`) still rely on `SlotTuple` encodings.

## Near-Term Direction

Likely next steps (see `futureideas.md`):

- richer literal syntax with compile-time shape checks
- more ergonomic lambda-free APIs
- extending dim algebra toward crop/resize/broadcast-style transforms
- more complete einsum semantics (diagonal/repeated labels, broader contraction rules)
