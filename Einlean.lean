/-
# Einlean

Compile-time verifiable tensor operations, einops style.

Dimensions are compile-time markers that carry a runtime size.
All position mappings (for rearrange, einsum) are resolved at
compile time via type classes — no runtime dim identity needed.
-/

namespace Einlean

-- ============================================
-- DIMENSIONS
-- ============================================

structure Dim where
  size : Nat
  deriving Repr, Inhabited

def dim (size : Nat) : Dim := ⟨size⟩

abbrev DimList := List Dim

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

def computeStrides (shape : Array Nat) : Array Nat := Id.run do
  let n := shape.size
  if n == 0 then return #[]
  let mut strides := Array.mkArray n 1
  let mut acc := 1
  for i in List.reverse (List.range n) do
    strides := strides.set! i acc
    acc := acc * shape[i]!
  return strides

def totalSize (shape : Array Nat) : Nat :=
  shape.foldl (· * ·) 1

def flatIndex (indices : Array Nat) (strides : Array Nat) (offset : Nat) : Nat := Id.run do
  let mut idx := offset
  for i in [:indices.size] do
    idx := idx + indices[i]! * strides[i]!
  return idx

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

def shapeOf (dims : DimList) : Array Nat :=
  dims.map (fun d => d.size) |>.toArray

-- ============================================
-- TENSORS
-- ============================================

structure Tensor (dims : DimList) where
  data : FloatArray
  shape : Array Nat
  strides : Array Nat
  offset : Nat := 0

def Tensor.ofFn {dims : DimList} (f : Array Nat → Float) : Tensor dims := Id.run do
  let shape := shapeOf dims
  let strides := computeStrides shape
  let total := totalSize shape
  let mut data := FloatArray.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat shape
    data := data.push (f idx)
  return { data := data, shape := shape, strides := strides, offset := 0 }

def Tensor.zeros {dims : DimList} : Tensor dims :=
  Tensor.ofFn (fun _ => 0.0)

def Tensor.fill {dims : DimList} (v : Float) : Tensor dims :=
  Tensor.ofFn (fun _ => v)

def Tensor.get! {dims : DimList} (t : Tensor dims) (indices : Array Nat) : Float :=
  let flat := flatIndex indices t.strides t.offset
  t.data.get! flat

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

structure Slot (dims : DimList) (i : Fin dims.length) where
  private mk :: unit : Unit

def SlotTuple : (dims : DimList) → Type
  | [] => PUnit
  | [d0] => Slot [d0] ⟨0, Nat.zero_lt_succ 0⟩
  | [d0, d1] => Slot [d0, d1] ⟨0, Nat.zero_lt_succ 1⟩ × Slot [d0, d1] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩
  | [d0, d1, d2] => Slot [d0, d1, d2] ⟨0, Nat.zero_lt_succ 2⟩ × Slot [d0, d1, d2] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 1)⟩ × Slot [d0, d1, d2] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))⟩
  | [d0, d1, d2, d3] => Slot [d0, d1, d2, d3] ⟨0, Nat.zero_lt_succ 3⟩ × Slot [d0, d1, d2, d3] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 2)⟩ × Slot [d0, d1, d2, d3] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1))⟩ × Slot [d0, d1, d2, d3] ⟨3, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0)))⟩
  | _ :: _ :: _ :: _ :: _ :: _ => PUnit

-- ============================================
-- REARRANGE (position-based)
-- ============================================

class HasDims (Out : Type) (dims : DimList) where
  outDims : DimList
  outPerm : List Nat

instance (dims : DimList) (i : Fin dims.length) : HasDims (Slot dims i) dims where
  outDims := [dims.get i]
  outPerm := [i.val]

instance (dims : DimList) {α β : Type} [HasDims α dims] [HasDims β dims] : HasDims (α × β) dims where
  outDims := (HasDims.outDims (Out := α) (dims := dims)) ++
    (HasDims.outDims (Out := β) (dims := dims))
  outPerm := (HasDims.outPerm (Out := α) (dims := dims)) ++
    (HasDims.outPerm (Out := β) (dims := dims))

def Tensor.rearrange {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    (t : Tensor dims) (_f : SlotTuple dims → Out) : Tensor h.outDims := Id.run do
  let perm := h.outPerm.toArray
  let n := perm.size
  let mut newShape := Array.mkArray n 0
  let mut newStrides := Array.mkArray n 0
  for i in [:n] do
    let srcIdx := perm[i]!
    newShape := newShape.set! i (t.shape[srcIdx]!)
    newStrides := newStrides.set! i (t.strides[srcIdx]!)
  return { data := t.data
           shape := newShape
           strides := newStrides
           offset := t.offset }

-- ============================================
-- EINSUM (position-based)
-- ============================================

class EinsumOut (Out : Type) (A B : DimList) where
  outDims : DimList
  /-- For each output position: (fromB, srcIdx).
      fromB=false → from A at srcIdx, fromB=true → from B at srcIdx. -/
  outSrc : List (Bool × Nat)

instance (A B : DimList) (i : Fin A.length) : EinsumOut (Slot A i) A B where
  outDims := [A.get i]
  outSrc := [(false, i.val)]

instance (A B : DimList) (i : Fin B.length) : EinsumOut (Slot B i) A B where
  outDims := [B.get i]
  outSrc := [(true, i.val)]

instance (A B : DimList) {α β : Type} [EinsumOut α A B] [EinsumOut β A B] : EinsumOut (α × β) A B where
  outDims := (EinsumOut.outDims (Out := α) (A := A) (B := B)) ++
    (EinsumOut.outDims (Out := β) (A := A) (B := B))
  outSrc := (EinsumOut.outSrc (Out := α) (A := A) (B := B)) ++
    (EinsumOut.outSrc (Out := β) (A := A) (B := B))

def Tensor.einsum {A B : DimList} {Out : Type}
    [h : EinsumOut Out A B]
    (x : Tensor A) (y : Tensor B)
    (_f : SlotTuple A → SlotTuple B → Out) : Tensor h.outDims := Id.run do
  let outSrc := h.outSrc.toArray

  -- Build output shape
  let outLen := outSrc.size
  let mut outShape := Array.mkArray outLen 0
  for oIdx in [:outLen] do
    let (fromB, srcIdx) := outSrc[oIdx]!
    outShape := outShape.set! oIdx (if fromB then y.shape[srcIdx]! else x.shape[srcIdx]!)

  -- Which positions of A / B appear in the output?
  let mut aIsOut := Array.mkArray A.length false
  let mut bIsOut := Array.mkArray B.length false
  let mut aToOut := Array.mkArray A.length 0
  let mut bToOut := Array.mkArray B.length 0
  for oIdx in [:outLen] do
    let (fromB, srcIdx) := outSrc[oIdx]!
    if fromB then
      bIsOut := bIsOut.set! srcIdx true
      bToOut := bToOut.set! srcIdx oIdx
    else
      aIsOut := aIsOut.set! srcIdx true
      aToOut := aToOut.set! srcIdx oIdx

  -- Contracted positions (those NOT in output), paired positionally
  let mut aContr : Array Nat := #[]
  for p in [:A.length] do
    if !(aIsOut[p]!) then aContr := aContr.push p
  let mut bContr : Array Nat := #[]
  for p in [:B.length] do
    if !(bIsOut[p]!) then bContr := bContr.push p

  -- Contracted shape (from A side)
  let contrShape := aContr.map (fun p => x.shape[p]!)

  -- Map: contracted A/B position → contracted loop index
  let mut aToContr := Array.mkArray A.length 0
  for cIdx in [:aContr.size] do
    aToContr := aToContr.set! (aContr[cIdx]!) cIdx
  let mut bToContr := Array.mkArray B.length 0
  for cIdx in [:bContr.size] do
    bToContr := bToContr.set! (bContr[cIdx]!) cIdx

  let outTotal := totalSize outShape
  let contrTotal := totalSize contrShape

  let mut resultData := FloatArray.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    let mut acc : Float := 0.0
    for contrFlat in [:contrTotal] do
      let contrIdx := toMultiIndex contrFlat contrShape
      -- Build A multi-index
      let mut aIdx := Array.mkArray A.length 0
      for p in [:A.length] do
        if aIsOut[p]! then
          aIdx := aIdx.set! p (outIdx[aToOut[p]!]!)
        else
          aIdx := aIdx.set! p (contrIdx[aToContr[p]!]!)
      -- Build B multi-index
      let mut bIdx := Array.mkArray B.length 0
      for p in [:B.length] do
        if bIsOut[p]! then
          bIdx := bIdx.set! p (outIdx[bToOut[p]!]!)
        else
          bIdx := bIdx.set! p (contrIdx[bToContr[p]!]!)
      acc := acc + x.get! aIdx * y.get! bIdx
    resultData := resultData.push acc
  return { data := resultData
           shape := outShape
           strides := computeStrides outShape
           offset := 0 }

-- ============================================
-- EXAMPLES
-- ============================================

def di := dim 2
def dj := dim 3

-- 2×3 matrix [[1,2,3],[4,5,6]]
def small : Tensor [di, dj] :=
  Tensor.ofFn fun idx => Float.ofNat (idx[0]! * 3 + idx[1]! + 1)

-- Transpose — same data, permuted strides
def smallT : Tensor [dj, di] :=
  small.rearrange fun (i, j) => (j, i)

def i := dim 2
def j := dim 4
def k := dim 3

-- 2×3 matrix [[1,2,3],[4,5,6]]
def a : Tensor [i, k] :=
  Tensor.ofFn fun idx => Float.ofNat (idx[0]! * 3 + idx[1]! + 1)

-- 3×4 matrix [[10,11,12,13],[14,15,16,17],[18,19,20,21]]
def bmat : Tensor [k, j] :=
  Tensor.ofFn fun idx => Float.ofNat (idx[0]! * 4 + idx[1]! + 10)

set_option linter.unusedVariables false in
def cmat : Tensor [i, j] :=
  Tensor.einsum a bmat (fun (i, k) (k, j) => (i, j))

-- Image — sizes live in the dims
def b := dim 32
def h := dim 224
def w := dim 224
def c := dim 3

def image : Tensor [b, h, w, c] := Tensor.zeros

def transposed : Tensor [b, c, h, w] :=
  image.rearrange fun (b, h, w, c) => (b, c, h, w)

#eval small.toList    -- [1, 2, 3, 4, 5, 6]
#eval smallT.toList   -- [1, 4, 2, 5, 3, 6]
#eval cmat.toList     -- [92, 98, 104, 110, 218, 233, 248, 263]

end Einlean
