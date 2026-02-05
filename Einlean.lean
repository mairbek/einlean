/-
# Einlean

Compile-time verifiable tensor operations, einops style.
-/

namespace Einlean

-- ============================================
-- DIMENSIONS
-- ============================================

/-- A dimension identifier with a runtime size. Identity is by `id` only. -/
structure Dim where
  private mk ::
    id : Nat
    size : Nat
  deriving Repr, Inhabited

instance : BEq Dim where
  beq a b := a.id == b.id

instance : Hashable Dim where
  hash d := hash d.id

/-- Create a dimension with a unique id and a runtime size. -/
def freshDim (id : Nat) (size : Nat) : Dim :=
  ⟨id, size⟩

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
-- HELPER FUNCTIONS
-- ============================================

/-- Find position of a Dim in a DimList. -/
def findDimIdx (dims : DimList) (d : Dim) : Option Nat :=
  let rec go (l : List Dim) (idx : Nat) : Option Nat :=
    match l with
    | [] => none
    | x :: xs => if x == d then some idx else go xs (idx + 1)
  go dims 0

/-- Compute row-major strides from a shape array. -/
def computeStrides (shape : Array Nat) : Array Nat := Id.run do
  let n := shape.size
  if n == 0 then return #[]
  let mut strides := Array.mkArray n 1
  let mut acc := 1
  for i in List.reverse (List.range n) do
    strides := strides.set! i acc
    acc := acc * shape[i]!
  return strides

/-- Total number of elements (product of shape). -/
def totalSize (shape : Array Nat) : Nat :=
  shape.foldl (· * ·) 1

/-- Multi-index to flat index using strides and offset. -/
def flatIndex (indices : Array Nat) (strides : Array Nat) (offset : Nat) : Nat := Id.run do
  let mut idx := offset
  for i in [:indices.size] do
    idx := idx + indices[i]! * strides[i]!
  return idx

/-- Flat index to multi-index (row-major, given shape). -/
def toMultiIndex (flat : Nat) (shape : Array Nat) : Array Nat := Id.run do
  let strides := computeStrides shape
  let n := shape.size
  let mut indices := Array.mkArray n 0
  let mut rem := flat
  for i in [:n] do
    let s := strides[i]!
    if s > 0 then
      indices := indices.set! i (rem / s)
      rem := rem % s
    else
      indices := indices.set! i 0
  return indices

/-- Extract shape from a DimList (sizes come from each Dim). -/
def shapeOf (dims : DimList) : Array Nat :=
  dims.map (fun d => d.size) |>.toArray

-- ============================================
-- TENSORS
-- ============================================

/-- A tensor with a statically known dimension signature. -/
structure Tensor (dims : DimList) where
  data : FloatArray
  shape : Array Nat
  strides : Array Nat
  offset : Nat := 0

/-- Create a tensor populated by an element function over multi-indices. -/
def Tensor.ofFn (dims : DimList) (f : Array Nat → Float) : Tensor dims := Id.run do
  let shape := shapeOf dims
  let strides := computeStrides shape
  let total := totalSize shape
  let mut data := FloatArray.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat shape
    data := data.push (f idx)
  return { data := data, shape := shape, strides := strides, offset := 0 }

/-- Create a tensor filled with zeros. -/
def Tensor.zeros (dims : DimList) : Tensor dims :=
  Tensor.ofFn dims (fun _ => 0.0)

/-- Element access via multi-index array. -/
def Tensor.get! {dims : DimList} (t : Tensor dims) (indices : Array Nat) : Float :=
  let flat := flatIndex indices t.strides t.offset
  t.data.get! flat

/-- Read all elements in logical (row-major) order. -/
def Tensor.toList {dims : DimList} (t : Tensor dims) : List Float := Id.run do
  let total := totalSize t.shape
  let mut result : Array Float := Array.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat t.shape
    let physFlat := flatIndex idx t.strides t.offset
    result := result.push (t.data.get! physFlat)
  return result.toList

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

/-- Rearrange a tensor by returning slots in a new order (stride permutation, no data copy). -/
def Tensor.rearrange {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    (t : Tensor dims) (_f : SlotTuple dims → Out) : Tensor h.outDims := Id.run do
  let outDims := h.outDims
  let n := outDims.length
  let mut newShape := Array.mkArray n 0
  let mut newStrides := Array.mkArray n 0
  for i in [:n] do
    let d := outDims[i]!
    match findDimIdx dims d with
    | some srcIdx =>
      newShape := newShape.set! i (t.shape[srcIdx]!)
      newStrides := newStrides.set! i (t.strides[srcIdx]!)
    | none =>
      newShape := newShape.set! i 0
      newStrides := newStrides.set! i 0
  return { data := t.data
           shape := newShape
           strides := newStrides
           offset := t.offset }

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

/-- Check if a Dim is in a DimList. -/
def dimIn (d : Dim) (ds : DimList) : Bool :=
  ds.any (· == d)

/-- Binary einsum driven by a slot lambda (nested loop contraction). -/
def Tensor.einsum {A B : DimList} {Out : Type}
    [h : EinsumOut Out A B]
    (x : Tensor A) (y : Tensor B)
    (_f : SlotTuple A → SlotTuple B → Out) : Tensor h.outDims := Id.run do
  let outDims := h.outDims
  -- Compute contracted dims: dims in both A and B but not in outDims
  let contracted := A.filter (fun d => dimIn d B && !dimIn d outDims)
  -- Build output shape
  let outShape := outDims.map (fun d =>
    match findDimIdx A d with
    | some idx => x.shape[idx]!
    | none =>
      match findDimIdx B d with
      | some idx => y.shape[idx]!
      | none => 0) |>.toArray
  -- Build contracted shape
  let contrShape := contracted.map (fun d =>
    match findDimIdx A d with
    | some idx => x.shape[idx]!
    | none => 0) |>.toArray
  let outTotal := totalSize outShape
  let contrTotal := totalSize contrShape
  let mut resultData := FloatArray.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    let mut acc : Float := 0.0
    for contrFlat in [:contrTotal] do
      let contrIdx := toMultiIndex contrFlat contrShape
      -- Build A's multi-index
      let mut aIdx := Array.mkArray A.length 0
      for ai in [:A.length] do
        let d := A[ai]!
        match findDimIdx outDims d with
        | some oi => aIdx := aIdx.set! ai (outIdx[oi]!)
        | none =>
          match findDimIdx contracted d with
          | some ci => aIdx := aIdx.set! ai (contrIdx[ci]!)
          | none => pure ()
      -- Build B's multi-index
      let mut bIdx := Array.mkArray B.length 0
      for bi in [:B.length] do
        let d := B[bi]!
        match findDimIdx outDims d with
        | some oi => bIdx := bIdx.set! bi (outIdx[oi]!)
        | none =>
          match findDimIdx contracted d with
          | some ci => bIdx := bIdx.set! bi (contrIdx[ci]!)
          | none => pure ()
      let aVal := x.get! aIdx
      let bVal := y.get! bIdx
      acc := acc + aVal * bVal
    resultData := resultData.push acc
  return { data := resultData
           shape := outShape
           strides := computeStrides outShape
           offset := 0 }

-- ============================================
-- EXAMPLES
-- ============================================

-- Dimensions carry their sizes
def di : Dim := freshDim 0 2
def dj : Dim := freshDim 1 3

-- 2×3 matrix: [[1,2,3],[4,5,6]]
def small : Tensor [di, dj] :=
  Tensor.ofFn [di, dj] (fun idx => Float.ofNat (idx[0]! * 3 + idx[1]! + 1))

-- Transpose — same data, permuted strides
def smallT : Tensor [dj, di] :=
  small.rearrange fun (i, j) => (j, i)

def i : Dim := freshDim 20 2
def j : Dim := freshDim 21 4
def k : Dim := freshDim 22 3

-- 2×3 matrix: [[1,2,3],[4,5,6]]
def a : Tensor [i, k] :=
  Tensor.ofFn [i, k] (fun idx => Float.ofNat (idx[0]! * 3 + idx[1]! + 1))

-- 3×4 matrix: [[10,11,12,13],[14,15,16,17],[18,19,20,21]]
def bmat : Tensor [k, j] :=
  Tensor.ofFn [k, j] (fun idx => Float.ofNat (idx[0]! * 4 + idx[1]! + 10))

set_option linter.unusedVariables false in
def cmat : Tensor [i, j] :=
  Tensor.einsum a bmat (fun (i, k) (k, j) => (i, j))

-- Image example — sizes live in the dims
def b : Dim := freshDim 10 32
def h : Dim := freshDim 11 224
def w : Dim := freshDim 12 224
def c : Dim := freshDim 13 3

def image : Tensor [b, h, w, c] := Tensor.zeros [b, h, w, c]

def transposed : Tensor [b, c, h, w] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)

#eval small.toList    -- [1, 2, 3, 4, 5, 6]
#eval smallT.toList   -- [1, 4, 2, 5, 3, 6]
#eval cmat.toList     -- [92, 98, 104, 110, 218, 233, 248, 263]

end Einlean
