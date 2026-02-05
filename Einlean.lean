/-
# Einlean

Compile-time verifiable tensor operations, einops style.
-/

namespace Einlean

-- ============================================
-- DIMENSIONS
-- ============================================

/-- A dimension identifier with a private constructor. -/
structure Dim where
  private mk :: id : Nat
  deriving DecidableEq, Repr, Hashable

/-- Create a dimension token (placeholder; uniqueness policy is external). -/
def freshDim (id : Nat) : Dim :=
  ⟨id⟩

/-- Type-level dimension list (shape signature). -/
abbrev DimList := List Dim

/-- Generate a list of Fin values from 0 to n-1. -/
def List.finRange (n : Nat) : List (Fin n) :=
  let rec go (i : Nat) (acc : List (Fin n)) : List (Fin n) :=
    if h : i < n then
      go (i + 1) (⟨i, h⟩ :: acc)
    else
      acc.reverse
  go 0 []

-- ============================================
-- TENSORS
-- ============================================

/-- A tensor with a statically known dimension signature.
    - `dims`: the list of dimensions (compile-time shape signature)
    - `data`: the underlying float array
    - `sizes`: runtime size for each dimension position -/
structure Tensor (dims : DimList) where
  data : FloatArray
  sizes : Dim → Option Nat

-- ============================================
-- SLOTS
-- ============================================

/-- A type-level marker for position `i` in `dims`. -/
structure Slot (dims : DimList) (i : Fin dims.length) where
  private mk :: unit : Unit

/-- Right-associated slot tuple for a dimension list.
    Defined by structural recursion on the list for clean type reduction. -/
def SlotTuple : (dims : DimList) → Type
  | [] => PUnit
  | [d0] => Slot [d0] ⟨0, Nat.zero_lt_succ 0⟩
  | [d0, d1] => Slot [d0, d1] ⟨0, Nat.zero_lt_succ 1⟩ × Slot [d0, d1] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩
  | [d0, d1, d2] => Slot [d0, d1, d2] ⟨0, Nat.zero_lt_succ 2⟩ × Slot [d0, d1, d2] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 1)⟩ × Slot [d0, d1, d2] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))⟩
  | [d0, d1, d2, d3] => Slot [d0, d1, d2, d3] ⟨0, Nat.zero_lt_succ 3⟩ × Slot [d0, d1, d2, d3] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 2)⟩ × Slot [d0, d1, d2, d3] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1))⟩ × Slot [d0, d1, d2, d3] ⟨3, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0)))⟩
  | _ :: _ :: _ :: _ :: _ :: _ => PUnit  -- fallback for larger lists

-- ============================================
-- DIM EXTRACTION
-- ============================================

/-- Extract output dimensions from a slot tuple type. -/
class HasDims (Out : Type) (dims : DimList) where
  outDims : DimList

instance (dims : DimList) (i : Fin dims.length) : HasDims (Slot dims i) dims where
  outDims := [dims.get i]

instance (dims : DimList) {α β : Type} [HasDims α dims] [HasDims β dims] : HasDims (α × β) dims where
  outDims := (HasDims.outDims (Out := α) (dims := dims)) ++
    (HasDims.outDims (Out := β) (dims := dims))

-- ============================================
-- REARRANGE
-- ============================================

-- ============================================
-- REARRANGE
-- ============================================

/-- Rearrange a tensor by returning slots in a new order. -/
def Tensor.rearrange {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    (t : Tensor dims) (f : SlotTuple dims → Out) : Tensor h.outDims :=
  { data := t.data
    sizes := t.sizes
  }

-- ============================================
-- EINSUM (BINARY)
-- ============================================

/-- Extract output dims for binary einsum. -/
class EinsumOut (Out : Type) (A B : DimList) where
  outDims : DimList

instance (A B : DimList) (i : Fin A.length) : EinsumOut (Slot A i) A B where
  outDims := [A.get i]

instance (A B : DimList) (i : Fin B.length) : EinsumOut (Slot B i) A B where
  outDims := [B.get i]

instance (A B : DimList) {α β : Type} [EinsumOut α A B] [EinsumOut β A B] : EinsumOut (α × β) A B where
  outDims := (EinsumOut.outDims (Out := α) (A := A) (B := B)) ++
    (EinsumOut.outDims (Out := β) (A := A) (B := B))

/-- Binary einsum driven by a slot lambda. -/
def Tensor.einsum {A B : DimList} {Out : Type}
    [h : EinsumOut Out A B]
    (x : Tensor A) (y : Tensor B)
    (f : SlotTuple A → SlotTuple B → Out) : Tensor h.outDims :=
  { data := x.data
    sizes := x.sizes
  }

-- ============================================
-- EXAMPLES
-- ============================================

def di : Dim := freshDim 0
def dj : Dim := freshDim 1
def dk : Dim := freshDim 2

def transpose_example (t : Tensor [di, dj]) : Tensor [dj, di] :=
  t.rearrange fun (i, j) => (j, i)

def b : Dim := freshDim 10
def h : Dim := freshDim 11
def w : Dim := freshDim 12
def c : Dim := freshDim 13

def image : Tensor [b, h, w, c] :=
  { data := FloatArray.mkEmpty 0
    sizes := fun d =>
      if d == b then some 32 else
      if d == h then some 224 else
      if d == w then some 224 else
      if d == c then some 3 else
      none
  }

def transposed : Tensor [b, c, h, w] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)

def i : Dim := freshDim 20
def j : Dim := freshDim 21
def k : Dim := freshDim 22

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

end Einlean
