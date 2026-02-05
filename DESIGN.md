# Einlean Design

Compile-time verifiable tensor operations in Lean 4, einops-style.

## Goal

```lean
def image : Tensor [batch, height, width, channels] := ...

-- Rearrange with compile-time verification
def transposed : Tensor [batch, channels, height, width] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)

-- Type error: output dims don't match declared type
def bad : Tensor [batch, batch, height, width] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)  -- won't compile
```

## Core Types

### Dimension

```lean
structure Dim where
  id : Nat
```

Unique identifier for each dimension axis.

### Tensor

```lean
structure Tensor (dims : List Dim) where
  data : FloatArray
  sizes : Fin dims.length → Nat
```

Tensor type is parameterized by its dimension list.

### Slot

```lean
structure Slot (dims : List Dim) (i : Fin dims.length)
```

A type-level marker representing position `i` in dimension list `dims`. Carries no runtime data—exists only for the type system.

Key property: `Slot dims i` knows both:
- Which dimension list it belongs to
- Which position it occupies

## The Rearrange Pattern

### Input: Slot Tuple

For a tensor with `dims = [d0, d1, d2]`, construct:

```lean
SlotTuple dims = Slot dims 0 × Slot dims 1 × Slot dims 2
```

User receives this as `(s0, s1, s2)` in their lambda.

### Output: Reordered Slots

User returns slots in new order:

```lean
fun (s0, s1, s2) => (s2, s0, s1)
```

Return type: `Slot dims 2 × Slot dims 0 × Slot dims 1`

### Extract Output Dims

A type class extracts the dimension list from the output type:

```lean
class HasDims (α : Type) (dims : List Dim) where
  toDims : List Dim

-- Base case: single slot
instance : HasDims (Slot dims i) dims where
  toDims := [dims.get i]

-- Inductive case: product
instance [HasDims α dims] [HasDims β dims] : HasDims (α × β) dims where
  toDims := HasDims.toDims α ++ HasDims.toDims β
```

For `Slot dims 2 × Slot dims 0 × Slot dims 1`, this yields `[d2, d0, d1]`.

### Rearrange Signature

```lean
def Tensor.rearrange
    {dims : List Dim}
    {Out : Type}
    [h : HasDims Out dims]
    (t : Tensor dims)
    (f : SlotTuple dims → Out)
    : Tensor h.toDims
```

## Type-Level Flow

```
Tensor [batch, height, width, channels]
           ↓
   SlotTuple [b, h, w, c]
           ↓
   fun (b, h, w, c) => (b, c, h, w)
           ↓
   Output type: Slot 0 × Slot 3 × Slot 1 × Slot 2
           ↓
   HasDims extracts: [batch, channels, height, width]
           ↓
Tensor [batch, channels, height, width]
```

## Examples

### 1. Define a Tensor

```lean
def b : Dim := freshDim 0
def h : Dim := freshDim 1
def w : Dim := freshDim 2
def c : Dim := freshDim 3

def image : Tensor [b, h, w, c] :=
  { data := FloatArray.mkEmpty 0
    sizes := fun d =>
      if d == b then some 32 else
      if d == h then some 224 else
      if d == w then some 224 else
      if d == c then some 3 else
      none
  }
```

### 2. Rearrange a Tensor

```lean
def transposed : Tensor [b, c, h, w] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)
```

### 3. Matmul via Einsum

```lean
def i : Dim := freshDim 10
def j : Dim := freshDim 11
def k : Dim := freshDim 12

def a : Tensor [i, k] :=
  { data := FloatArray.mkEmpty 0
    sizes := fun d =>
      if d == i then some 2 else
      if d == k then some 3 else
      none
  }

def bmat : Tensor [k, j] :=
  { data := FloatArray.mkEmpty 0
    sizes := fun d =>
      if d == k then some 3 else
      if d == j then some 4 else
      none
  }

def cmat : Tensor [i, j] :=
  Tensor.einsum a bmat (fun (i, k) (k, j) => (i, j))
```

## Implementation Tasks

1. **Define `Slot`** - zero-sized type with position index
2. **Define `SlotTuple`** - product type for n slots
3. **Build slot values** - construct the tuple to pass to user's lambda
4. **Define `HasDims`** - type class to extract dims from output
5. **Define `Tensor.rearrange`** - tie it together

## Future Extensions

- **Reduce**: `fun (b, h, w, c) => (b, c)` — sum over h, w
- **Split**: `fun (b, hw, c) => (b, h, w, c)` — split dimension
- **Merge**: `fun (b, h, w, c) => (b, hw, c)` — merge dimensions
- **Broadcast**: introduce new dimensions
- **Einsum**: `a[i,j] * b[j,k] → c[i,k]`
