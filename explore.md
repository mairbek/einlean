# Einlean Explore

Design sketch for compile-time verified tensor rearrange and binary einsum in Lean 4.

## Goal

```lean
def image : Tensor [batch, height, width, channels] := ...

def transposed : Tensor [batch, channels, height, width] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)

-- Type error: output dims don't match declared type
def bad : Tensor [batch, batch, height, width] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)
```

## Core Types

```lean
opaque Dim : Type

structure Tensor (dims : List Dim) where
  data  : FloatArray
  sizes : Dim → Option Nat
  sizes_wf : ∀ d, (sizes d = none) ↔ d ∉ dims

structure Slot (dims : List Dim) (i : Fin dims.length) where
  private mk :: unit : Unit
```

Slots are not forgeable; only trusted code can construct them.

## Slot Tuples

Right-associated tuple shape so `(b, h, w, c)` works in lambdas.

```lean
def SlotTuple : List Dim → Type
| []      => PUnit
| d :: ds => Slot (d :: ds) ⟨0, by decide⟩ × SlotTuple ds
```

## Rearrange (Permutation Only)

Rearrange only permits permutations: no duplicates and no omissions.

```lean
structure IsPermutation (Out : Type) (dims : List Dim) where
  outDims : List Dim
  perm    : outDims ~ dims

def Tensor.rearrange
  {dims : List Dim} {Out : Type}
  [p : IsPermutation Out dims]
  (t : Tensor dims)
  (f : SlotTuple dims → Out)
  : Tensor p.outDims
```

Instances for `IsPermutation` are closed to prevent user-defined cheating.

## Einsum (Binary, Lambda-Based)

Binary einsum with explicit lambda, allowing outer products and contractions.
Output dims must be nodup.

```lean
structure EinsumSpec (Out : Type) (A B : List Dim) where
  outDims     : List Dim
  nodup_out   : List.Nodup outDims
  out_subset  : ∀ d, d ∈ outDims → d ∈ A ∨ d ∈ B
  shared_contract : ∀ d, d ∈ A → d ∈ B → d ∉ outDims → True

def Tensor.einsum
  {A B : List Dim} {Out : Type}
  [p : EinsumSpec Out A B]
  (x : Tensor A) (y : Tensor B)
  (f : SlotTuple A → SlotTuple B → Out)
  : Tensor p.outDims
```

Examples

```lean
-- Matrix multiply
def c : Tensor [i, j] :=
  Tensor.einsum a b (fun (i, k) (k, j) => (i, j))

-- Batch matrix multiply
def c : Tensor [b, i, j] :=
  Tensor.einsum a b (fun (b, i, k) (b, k, j) => (b, i, j))

-- Attention scores
def s : Tensor [b, q, k] :=
  Tensor.einsum q k (fun (b, q, d) (b, k, d) => (b, q, k))

-- Weighted sum
def o : Tensor [b, q, d] :=
  Tensor.einsum w v (fun (b, q, k) (b, k, d) => (b, q, d))

-- Outer product
def op : Tensor [i, j] :=
  Tensor.einsum a b (fun (i) (j) => (i, j))
```

## Safety Boundaries

- `Dim` and `Slot` constructors are private.
- `IsPermutation` and `EinsumSpec` instances are closed.
- `sizes` is keyed by `Dim`, so `rearrange` preserves sizes unchanged.

## Future Extensions

- Diagonal/repeated labels in einsum.
- Broadcasting and explicit reductions.
- Matmul alias for common einsum patterns.
