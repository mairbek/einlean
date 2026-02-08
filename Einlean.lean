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

structure Atom where
  id : Nat
  size : Nat
  deriving Repr, Inhabited, BEq, DecidableEq

structure Dim where
  atoms : List Atom
  deriving Repr, Inhabited, BEq, DecidableEq

def dim (id : Nat) (size : Nat) : Dim := ⟨[⟨id, size⟩]⟩

open Lean in
scoped macro "dim!" size:term : term => do
  let some pos := (← getRef).getPos? | Macro.throwError "dim!: no source position"
  `(dim $(quote pos.byteIdx) $size)

def Dim.size (d : Dim) : Nat :=
  d.atoms.foldl (fun acc a => acc * a.size) 1

instance : Mul Dim where
  mul d1 d2 := ⟨d1.atoms ++ d2.atoms⟩

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

structure Tensor (dims : DimList) (α : Type := Int) where
  data : Array α
  shape : Array Nat
  strides : Array Nat
  offset : Nat := 0

def Tensor.ofFn {dims : DimList} {α : Type} (f : Array Nat → α) : Tensor dims α := Id.run do
  let shape := shapeOf dims
  let strides := computeStrides shape
  let total := totalSize shape
  let mut data : Array α := Array.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat shape
    data := data.push (f idx)
  return { data := data, shape := shape, strides := strides, offset := 0 }

def Tensor.zeros {dims : DimList} {α : Type} [Zero α] : Tensor dims α :=
  Tensor.ofFn (fun _ => 0)

def Tensor.fill {dims : DimList} {α : Type} (v : α) : Tensor dims α :=
  Tensor.ofFn (fun _ => v)

def Tensor.arange {dims : DimList} {α : Type} [Add α] [OfNat α 0] [OfNat α 1]
    (start : α := 0) (step : α := 1) : Tensor dims α := Id.run do
  let shape := shapeOf dims
  let strides := computeStrides shape
  let total := totalSize shape
  let mut data : Array α := Array.mkEmpty total
  let mut cur := start
  for flat in [:total] do
    let _ := flat
    data := data.push cur
    cur := cur + step
  return { data := data, shape := shape, strides := strides, offset := 0 }

def Tensor.arange1d {α : Type} [Add α] [OfNat α 0] [OfNat α 1]
    (d : Dim) (start : α := 0) (step : α := 1) : Tensor [d] α :=
  Tensor.arange (dims := [d]) start step

def arange {dims : DimList} {α : Type} [Add α] [OfNat α 0] [OfNat α 1]
    (start : α := 0) (step : α := 1) : Tensor dims α :=
  Tensor.arange (dims := dims) start step

def arange1d {α : Type} [Add α] [OfNat α 0] [OfNat α 1]
    (d : Dim) (start : α := 0) (step : α := 1) : Tensor [d] α :=
  Tensor.arange1d d start step

def Tensor.ofData {dims : DimList} {α : Type} (data : List α) : Tensor dims α :=
  let shape := shapeOf dims
  let strides := computeStrides shape
  let fa := data.foldl (fun acc v => acc.push v) (Array.mkEmpty data.length)
  { data := fa, shape := shape, strides := strides, offset := 0 }

private def flattenNested2 {α : Type} (rows : List (List α)) : List α :=
  rows.foldl (fun acc row => acc ++ row) []

private def flattenNested3 {α : Type} (xss : List (List (List α))) : List α :=
  xss.foldl (fun acc xs => acc ++ flattenNested2 xs) []

instance {d : Dim} : Coe (List Float) (Tensor [d] Float) where
  coe data := Tensor.ofData data

instance {d0 d1 : Dim} : Coe (List (List Float)) (Tensor [d0, d1] Float) where
  coe data := Tensor.ofData (flattenNested2 data)

instance {d0 d1 d2 : Dim} : Coe (List (List (List Float))) (Tensor [d0, d1, d2] Float) where
  coe data := Tensor.ofData (flattenNested3 data)

instance {d : Dim} : Coe (List Nat) (Tensor [d]) where
  coe data := Tensor.ofData (data.map Int.ofNat)

instance {d0 d1 : Dim} : Coe (List (List Nat)) (Tensor [d0, d1]) where
  coe data := Tensor.ofData (flattenNested2 (data.map (·.map Int.ofNat)))

instance {d0 d1 d2 : Dim} : Coe (List (List (List Nat))) (Tensor [d0, d1, d2]) where
  coe data := Tensor.ofData (flattenNested3 (data.map (·.map (·.map Int.ofNat))))

def Tensor.get! {dims : DimList} {α : Type} [Inhabited α] (t : Tensor dims α) (indices : Array Nat) : α :=
  let flat := flatIndex indices t.strides t.offset
  t.data.get! flat

def Tensor.toList {dims : DimList} {α : Type} [Inhabited α] (t : Tensor dims α) : List α := Id.run do
  let total := totalSize t.shape
  let mut result : Array α := Array.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat t.shape
    let physFlat := flatIndex idx t.strides t.offset
    result := result.push (t.data.get! physFlat)
  return result.toList

-- ============================================
-- TENSOR FORMATTING
-- ============================================

private def formatElem {α : Type} [ToString α] (x : α) : String :=
  toString x

private partial def formatSubtensor {α : Type} [ToString α] [Inhabited α]
    (data : Array α) (offset : Nat) (shape : List Nat) (indent : Nat) : String × Nat :=
  match shape with
  | [] => (formatElem (data.get! offset), offset + 1)
  | [n] => Id.run do
    let mut s := "["
    for idx in [:n] do
      if idx > 0 then s := s ++ ", "
      s := s ++ formatElem (data.get! (offset + idx))
    return (s ++ "]", offset + n)
  | n :: rest => Id.run do
    let innerSize := rest.foldl (· * ·) 1
    let mut s := "["
    let innerIndent := indent + 1
    for idx in [:n] do
      if idx > 0 then
        s := s ++ ",\n"
        for _ in [:innerIndent] do s := s ++ " "
      let (sub, _) := formatSubtensor data (offset + idx * innerSize) rest innerIndent
      s := s ++ sub
    return (s ++ "]", offset + n * innerSize)

def Tensor.format {dims : DimList} {α : Type} [ToString α] [Inhabited α] (t : Tensor dims α) : String :=
  let data := t.toList.toArray
  let (s, _) := formatSubtensor data 0 t.shape.toList 0
  s

instance {dims : DimList} {α : Type} [ToString α] [Inhabited α] : Repr (Tensor dims α) where
  reprPrec t _ := .text t.format

instance {dims : DimList} {α : Type} [ToString α] [Inhabited α] : ToString (Tensor dims α) where
  toString := Tensor.format

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
-- MERGED SLOTS (for dimension merging in rearrange)
-- ============================================

structure MergedSlot (dims : DimList) (is : List (Fin dims.length)) where
  private mk :: unit : Unit

instance (dims : DimList) (i j : Fin dims.length) :
    HMul (Slot dims i) (Slot dims j) (MergedSlot dims [i, j]) where
  hMul _ _ := ⟨()⟩

instance (dims : DimList) (is : List (Fin dims.length)) (j : Fin dims.length) :
    HMul (MergedSlot dims is) (Slot dims j) (MergedSlot dims (is ++ [j])) where
  hMul _ _ := ⟨()⟩

def mergedDim (dims : DimList) (is : List (Fin dims.length)) : Dim :=
  ⟨is.foldl (fun acc i => acc ++ (dims.get i).atoms) []⟩

-- ============================================
-- REARRANGE (position-based, supports merging)
-- ============================================

class HasDims (Out : Type) (dims : DimList) where
  outDims : DimList
  outMapping : List (List Nat)

instance (dims : DimList) (i : Fin dims.length) : HasDims (Slot dims i) dims where
  outDims := [dims.get i]
  outMapping := [[i.val]]

instance (dims : DimList) (is : List (Fin dims.length)) : HasDims (MergedSlot dims is) dims where
  outDims := [mergedDim dims is]
  outMapping := [is.map Fin.val]

instance (dims : DimList) {α β : Type} [HasDims α dims] [HasDims β dims] : HasDims (α × β) dims where
  outDims := (HasDims.outDims (Out := α) (dims := dims)) ++
    (HasDims.outDims (Out := β) (dims := dims))
  outMapping := (HasDims.outMapping (Out := α) (dims := dims)) ++
    (HasDims.outMapping (Out := β) (dims := dims))

def Tensor.rearrangeCore {inDims outDims : DimList} {α : Type} [Inhabited α]
    (t : Tensor inDims α) (mapping : Array (Array Nat)) : Tensor outDims α := Id.run do
  let outLen := mapping.size
  -- Build output shape: product of input sizes for each output axis
  let mut outShape := Array.mkArray outLen 0
  for oAxis in [:outLen] do
    let axes := mapping[oAxis]!
    let mut sz := 1
    for aIdx in [:axes.size] do
      sz := sz * t.shape[axes[aIdx]!]!
    outShape := outShape.set! oAxis sz
  let outStrides := computeStrides outShape
  let outTotal := totalSize outShape
  let mut resultData : Array α := Array.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    -- Build input multi-index by decomposing each output index
    let mut inIdx := Array.mkArray inDims.length 0
    for oAxis in [:outLen] do
      let axes := mapping[oAxis]!
      let mut remainder := outIdx[oAxis]!
      -- Decompose right-to-left (last merged axis varies fastest)
      for rev in [:axes.size] do
        let aIdx := axes.size - 1 - rev
        let p := axes[aIdx]!
        let sz := t.shape[p]!
        inIdx := inIdx.set! p (remainder % sz)
        remainder := remainder / sz
    let flat := flatIndex inIdx t.strides t.offset
    resultData := resultData.push (t.data.get! flat)
  return { data := resultData
           shape := outShape
           strides := outStrides
           offset := 0 }

def Tensor.rearrangeBy {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    {α : Type} [Inhabited α] (t : Tensor dims α) (_f : SlotTuple dims → Out) : Tensor h.outDims α :=
  Tensor.rearrangeCore t (h.outMapping.map (·.toArray) |>.toArray)

-- ============================================
-- LAMBDA-FREE REARRANGE (atom-based)
-- ============================================

def allAtoms (dims : DimList) : List Atom :=
  dims.foldl (fun acc d => acc ++ d.atoms) []

def validRearrange (inDims outDims : DimList) : Bool :=
  let inA := allAtoms inDims
  let outA := allAtoms outDims
  inA.length == outA.length &&
  inA.all (fun a => outA.any (· == a)) &&
  outA.all (fun a => inA.any (· == a))

def validReshape (inDims outDims : DimList) : Bool :=
  allAtoms inDims == allAtoms outDims

/-- For each output dim, find which input axis indices contribute to it by matching atom IDs. -/
def computeAtomMapping (inDims outDims : DimList) : Array (Array Nat) := Id.run do
  -- Build flat list of (inputAxisIndex, atom) pairs
  let mut inAtomIndex : Array (Nat × Atom) := #[]
  for axIdx in [:inDims.length] do
    let d := inDims[axIdx]!
    for a in d.atoms do
      inAtomIndex := inAtomIndex.push (axIdx, a)
  -- For each output dim, find matching input axes
  let mut result : Array (Array Nat) := #[]
  for outD in outDims do
    let mut axes : Array Nat := #[]
    for outAtom in outD.atoms do
      -- Find the input atom with matching id
      for (axIdx, inAtom) in inAtomIndex do
        if inAtom == outAtom then
          -- Only add axis if not already present
          if !(axes.any (· == axIdx)) then
            axes := axes.push axIdx
    result := result.push axes
  return result

def Tensor.rearrange {inDims outDims : DimList} {α : Type} [Inhabited α] (t : Tensor inDims α)
    (_valid : validRearrange inDims outDims = true := by decide) : Tensor outDims α :=
  Tensor.rearrangeCore t (computeAtomMapping inDims outDims)

def Tensor.reshape {inDims outDims : DimList} {α : Type} (t : Tensor inDims α)
    (_valid : validReshape inDims outDims = true := by decide) : Tensor outDims α :=
  let outShape := shapeOf outDims
  { data := t.data
    shape := outShape
    strides := computeStrides outShape
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

def Tensor.einsum {A B : DimList} {Out : Type} {α : Type}
    [h : EinsumOut Out A B]
    [Inhabited α] [Zero α] [Add α] [Mul α]
    (x : Tensor A α) (y : Tensor B α)
    (_f : SlotTuple A → SlotTuple B → Out) : Tensor h.outDims α := Id.run do
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

  let mut resultData : Array α := Array.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    let mut acc : α := Zero.zero
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

def di := dim! 2
def dj := dim! 3

-- 2×3 matrix [[1,2,3],[4,5,6]]
def small : Tensor [di, dj] := arange 1

-- Transpose (lambda-based) — same data, permuted strides
def smallT : Tensor [dj, di] :=
  small.rearrangeBy fun (i, j) => (j, i)

def i := dim! 2
def j := dim! 4
def k := dim! 3

-- 2×3 matrix [[1,2,3],[4,5,6]]
def a : Tensor [i, k] := arange 1

-- 3×4 matrix [[10,11,12,13],[14,15,16,17],[18,19,20,21]]
def bmat : Tensor [k, j] := arange 10

set_option linter.unusedVariables false in
def cmat : Tensor [i, j] :=
  Tensor.einsum a bmat (fun (i, _) (_, j) => (i, j))

-- Image — sizes live in the dims
def b := dim! 32
def h := dim! 224
def w := dim! 224
def c := dim! 3

def image : Tensor [b, h, w, c] := Tensor.zeros

-- Lambda-based (renamed to rearrangeBy)
def transposed : Tensor [b, c, h, w] :=
  image.rearrangeBy fun (b, h, w, c) => (b, c, h, w)

-- Merge dims (lambda-based): "b h w c -> h (b w) c"
def merged : Tensor [h, b * w, c] :=
  image.rearrangeBy fun (b, h, w, c) => (h, b * w, c)

def bw := b * w

def transposed2 : Tensor [b, c, h, w] := image.rearrange
def merged2 : Tensor [h, bw, c] := image.rearrange

-- Small merge test: 2×3 -> 6 (flatten)
#eval small      -- [[1, 2, 3], [4, 5, 6]]
#eval smallT     -- [[1, 4], [2, 5], [3, 6]]
#eval cmat       -- [[92, 98, 104, 110], [218, 233, 248, 263]]

def db := dim! 2
def dw := dim! 3
def example2d : Tensor [db, dw] := arange 1
#eval example2d     -- [[1, 2, 3], [4, 5, 6]]
#eval example2d.rearrangeBy fun (b, w) => b * w -- [1, 2, 3, 4, 5, 6]
#eval example2d.rearrangeBy fun (b, w) => w * b -- [1, 4, 2, 5, 3, 6]


def dflat := di * dj
def flatRange : Tensor [dflat] := arange
def reshapedRange : Tensor [di, dj] := flatRange.reshape
def reshapedRange2 : Tensor [di, dj] := (arange (dims := [di * dj])).reshape
def targetDirect : Tensor [di, dj] := arange
def targetDirectStep : Tensor [di, dj] Float := arange 10.0 0.5

-- This would fail (different atom identities):
-- def badDx := dim! 6
-- def badFlat : Tensor [badDx] := arange1d badDx
-- def badTarget : Tensor [di, dj] := badFlat.reshape

#eval flatRange      -- [0, 1, 2, 3, 4, 5]
#eval reshapedRange  -- [[0, 1, 2], [3, 4, 5]]
#eval reshapedRange2 -- [[0, 1, 2], [3, 4, 5]]
#eval targetDirect   -- [[0, 1, 2], [3, 4, 5]]
#eval targetDirectStep -- [[10, 10.500000, 11], [11.500000, 12, 12.500000]]


-- Lambda-free transpose
def smallT2 : Tensor [dj, di] := small.rearrange
#eval smallT2    -- [[1, 4], [2, 5], [3, 6]]

end Einlean
