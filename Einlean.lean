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
  rank : Nat := 0
  deriving Repr, Inhabited, BEq, DecidableEq

structure Dim where
  atoms : List Atom
  deriving Repr, Inhabited, BEq, DecidableEq

def dim (id : Nat) (size : Nat) : Dim := ⟨[{ id := id, size := size, rank := 0 }]⟩

open Lean in
scoped macro "dim!" size:term : term => do
  let some pos := (← getRef).getPos? | Macro.throwError "dim!: no source position"
  `(dim $(quote pos.byteIdx) $size)

def Dim.size (d : Dim) : Nat :=
  d.atoms.foldl (fun acc a => acc * a.size) 1

private def normalizeAtoms (atoms : List Atom) : List Atom :=
  let step (revAcc : List Atom) (a : Atom) : List Atom :=
    match revAcc with
    | [] => [a]
    | prev :: rest =>
        if prev.id == a.id then
          { id := prev.id, size := prev.size * a.size, rank := 0 } :: rest
        else
          a :: revAcc
  (atoms.foldl step []).reverse

instance : Mul Dim where
  mul d1 d2 := ⟨normalizeAtoms (d1.atoms ++ d2.atoms)⟩

def Dim.factorAt (d : Dim) (k tag : Nat)
    (_single : d.atoms.length = 1 := by decide)
    (_pos : k > 0 := by decide)
    (_div : d.size % k = 0 := by decide) : Dim :=
  let a := d.atoms.get! 0
  ⟨[{ id := a.id, size := k, rank := tag + 1 }]⟩

def Dim.factor! (d : Dim) (k : Nat)
    (_single : d.atoms.length = 1 := by decide)
    (_pos : k > 0 := by decide)
    (_div : d.size % k = 0 := by decide) : Dim :=
  Dim.factorAt d k k _single _pos _div

def Dim.factorPair! (d : Dim) (k : Nat)
    (_single : d.atoms.length = 1 := by decide)
    (_pos : k > 0 := by decide)
    (_div : d.size % k = 0 := by decide) : Dim × Dim :=
  let a := d.atoms.get! 0
  -- Keep outer rank > inner rank so atom-order reconstruction matches
  -- the common split pattern (outer, inner).
  let outer : Dim := ⟨[{ id := a.id, size := d.size / k, rank := 2 }]⟩
  let inner : Dim := ⟨[{ id := a.id, size := k, rank := 1 }]⟩
  (outer, inner)

open Lean in
syntax "factor! " ident "," ident " := " term "," term : command

open Lean in
macro_rules
  | `(factor! $outer:ident, $inner:ident := $d:term, $k:term) =>
      `(def $outer : Dim := (Dim.factorPair! $d $k).1
        def $inner : Dim := (Dim.factorPair! $d $k).2)

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

instance {dims : DimList} {α : Type} [Inhabited α] : Inhabited (Tensor dims α) where
  default := { data := #[], shape := #[], strides := #[], offset := 0 }

def Tensor.ofFn {dims : DimList} {α : Type} (f : Array Nat → α) : Tensor dims α := Id.run do
  let shape := shapeOf dims
  let strides := computeStrides shape
  let total := totalSize shape
  let mut data : Array α := Array.mkEmpty total
  for flat in [:total] do
    let idx := toMultiIndex flat shape
    data := data.push (f idx)
  return { data := data, shape := shape, strides := strides, offset := 0 }

def Tensor.ofArray {dims : DimList} {α : Type} (data : Array α) : Tensor dims α :=
  let shape := shapeOf dims
  let strides := computeStrides shape
  { data := data, shape := shape, strides := strides, offset := 0 }

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
  let fa := data.foldl (fun acc v => acc.push v) (Array.mkEmpty data.length)
  Tensor.ofArray (dims := dims) fa

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

def Tensor.slice0 {d : Dim} {ds : DimList} {α : Type}
    [Inhabited α] (t : Tensor (d :: ds) α) (i : Nat) : Tensor ds α :=
  if i < t.shape[0]! then
    let outShape := t.shape.extract 1 t.shape.size
    let outStrides := t.strides.extract 1 t.strides.size
    { data := t.data
      shape := outShape
      strides := outStrides
      offset := t.offset + i * t.strides[0]! }
  else
    panic! s!"Tensor index out of bounds: index {i}, size {t.shape[0]!}"

instance {d : Dim} {ds : DimList} {α : Type} [Inhabited α] :
    GetElem (Tensor (d :: ds) α) Nat (Tensor ds α) (fun _ i => i < i + 1) where
  getElem t i _ := t.slice0 i

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
-- REDUCE OPS
-- ============================================

inductive ReduceOp where
  | sum | mean | max | min
  deriving Repr, BEq

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
  | [d0, d1, d2, d3, d4] =>
      Slot [d0, d1, d2, d3, d4] ⟨0, Nat.zero_lt_succ 4⟩ ×
      Slot [d0, d1, d2, d3, d4] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 3)⟩ ×
      Slot [d0, d1, d2, d3, d4] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 2))⟩ ×
      Slot [d0, d1, d2, d3, d4] ⟨3, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1)))⟩ ×
      Slot [d0, d1, d2, d3, d4] ⟨4, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))))⟩
  | [d0, d1, d2, d3, d4, d5] =>
      Slot [d0, d1, d2, d3, d4, d5] ⟨0, Nat.zero_lt_succ 5⟩ ×
      Slot [d0, d1, d2, d3, d4, d5] ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 4)⟩ ×
      Slot [d0, d1, d2, d3, d4, d5] ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 3))⟩ ×
      Slot [d0, d1, d2, d3, d4, d5] ⟨3, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 2)))⟩ ×
      Slot [d0, d1, d2, d3, d4, d5] ⟨4, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 1))))⟩ ×
      Slot [d0, d1, d2, d3, d4, d5] ⟨5, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0)))))⟩
  | _ :: _ :: _ :: _ :: _ :: _ :: _ => PUnit

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

instance (dims : DimList) : HasDims PUnit dims where
  outDims := []
  outMapping := []

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

private def uniqueIds (atoms : List Atom) : List Nat :=
  atoms.foldl (fun ids a => if ids.any (· == a.id) then ids else ids ++ [a.id]) []

private def productForId (atoms : List Atom) (id : Nat) : Nat :=
  atoms.foldl (fun acc a => if a.id == id then acc * a.size else acc) 1

def validRearrange (inDims outDims : DimList) : Bool :=
  let inA := allAtoms inDims
  let outA := allAtoms outDims
  let inIds := uniqueIds inA
  let outIds := uniqueIds outA
  inIds.length == outIds.length &&
  inIds.all (fun id => outIds.any (· == id)) &&
  outIds.all (fun id => inIds.any (· == id)) &&
  inIds.all (fun id => productForId inA id == productForId outA id)

def validReshape (inDims outDims : DimList) : Bool :=
  allAtoms inDims == allAtoms outDims

def validReduce (inDims outDims : DimList) : Bool :=
  let inA := allAtoms inDims
  let outA := allAtoms outDims
  let outIds := uniqueIds outA
  let inIds := uniqueIds inA
  outIds.all (fun id => inIds.any (· == id)) &&
  outIds.all (fun id => productForId outA id == productForId inA id)

private def uniqueDims (dims : DimList) : DimList :=
  dims.foldl (fun acc d => if acc.any (· == d) then acc else acc ++ [d]) []

private def findDimIndex? (dims : Array Dim) (d : Dim) : Option Nat := Id.run do
  for i in [:dims.size] do
    if dims[i]! == d then
      return some i
  return none


def validEinsum2 (aDims bDims outDims : DimList) : Bool :=
  let outU := uniqueDims outDims
  let inU := uniqueDims (aDims ++ bDims)
  outU.length == outDims.length &&
  outU.all (fun d => inU.any (· == d))

def validEinsum3 (aDims bDims cDims outDims : DimList) : Bool :=
  let outU := uniqueDims outDims
  let inU := uniqueDims (aDims ++ bDims ++ cDims)
  outU.length == outDims.length &&
  outU.all (fun d => inU.any (· == d))

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

/-- Returns (outMapping, reducedAxes) for a reduce operation.
    outMapping: for each output dim, which input axes map to it.
    reducedAxes: input axes not present in any output dim. -/
def computeReduceInfo (inDims outDims : DimList) : Array (Array Nat) × Array Nat := Id.run do
  let mapping := computeAtomMapping inDims outDims
  let mut usedAxes : Array Bool := Array.mkArray inDims.length false
  for oIdx in [:mapping.size] do
    let axes := mapping[oIdx]!
    for aIdx in [:axes.size] do
      usedAxes := usedAxes.set! (axes[aIdx]!) true
  let mut reduced : Array Nat := #[]
  for p in [:inDims.length] do
    if !(usedAxes[p]!) then
      reduced := reduced.push p
  return (mapping, reduced)

private def atomDigitsFromAxisIndex (idx : Nat) (atomSizes : Array Nat) : Array Nat := Id.run do
  let n := atomSizes.size
  let mut out := Array.mkArray n 0
  let mut rem := idx
  for rev in [:n] do
    let j := n - 1 - rev
    let s := atomSizes[j]!
    if s > 0 then
      out := out.set! j (rem % s)
      rem := rem / s
    else
      out := out.set! j 0
  return out

private def axisIndexFromAtomDigits (digits atomSizes : Array Nat) : Nat := Id.run do
  let mut acc := 0
  for i in [:digits.size] do
    acc := acc * atomSizes[i]! + digits[i]!
  return acc

private def positionsForId (atoms : Array Atom) (id : Nat) : Array Nat := Id.run do
  let mut ps : Array Nat := #[]
  for i in [:atoms.size] do
    if atoms[i]!.id == id then
      ps := ps.push i
  return ps

private def sortPositionsByRank (atoms : Array Atom) (ps : Array Nat) : Array Nat :=
  let rec insert (p : Nat) (acc : List Nat) : List Nat :=
    match acc with
    | [] => [p]
    | q :: qs =>
        if atoms[p]!.rank >= atoms[q]!.rank then
          p :: acc
        else
          q :: insert p qs
  (ps.toList.foldl (fun acc p => insert p acc) []).toArray

private structure RootSpec where
  id : Nat
  inPos : Array Nat
  outPos : Array Nat
  inSizes : Array Nat
  outSizes : Array Nat

private def buildRootSpecs (inAtoms outAtoms : Array Atom) : Array RootSpec := Id.run do
  let mut ids : Array Nat := #[]
  for a in inAtoms do
    if !(ids.any (· == a.id)) then
      ids := ids.push a.id
  let mut specs : Array RootSpec := #[]
  for id in ids do
    let inPos := sortPositionsByRank inAtoms (positionsForId inAtoms id)
    let outPos := sortPositionsByRank outAtoms (positionsForId outAtoms id)
    let inSizes := inPos.map (fun p => inAtoms[p]!.size)
    let outSizes := outPos.map (fun p => outAtoms[p]!.size)
    specs := specs.push { id := id, inPos := inPos, outPos := outPos, inSizes := inSizes, outSizes := outSizes }
  return specs

private def Tensor.rearrangeByRoots {inDims outDims : DimList} {α : Type} [Inhabited α]
    (t : Tensor inDims α) : Tensor outDims α := Id.run do
  let outShape := shapeOf outDims
  let outTotal := totalSize outShape
  let inAtoms := (allAtoms inDims).toArray
  let outAtoms := (allAtoms outDims).toArray
  let rootSpecs := buildRootSpecs inAtoms outAtoms

  let mut inAtomStarts : Array Nat := Array.mkArray inDims.length 0
  let mut cur := 0
  for i in [:inDims.length] do
    inAtomStarts := inAtomStarts.set! i cur
    cur := cur + inDims[i]!.atoms.length

  let mut resultData : Array α := Array.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape

    let mut outAtomDigits : Array Nat := Array.mkEmpty outAtoms.size
    for ax in [:outDims.length] do
      let d := outDims[ax]!
      let sizes := d.atoms.map (·.size) |>.toArray
      let digs := atomDigitsFromAxisIndex outIdx[ax]! sizes
      for j in [:digs.size] do
        outAtomDigits := outAtomDigits.push digs[j]!

    let mut inAtomDigits := Array.mkArray inAtoms.size 0
    for spec in rootSpecs do
      let mut coord := 0
      for j in [:spec.outPos.size] do
        coord := coord * spec.outSizes[j]! + outAtomDigits[spec.outPos[j]!]!

      let mut rem := coord
      for rev in [:spec.inPos.size] do
        let j := spec.inPos.size - 1 - rev
        let s := spec.inSizes[j]!
        if s > 0 then
          inAtomDigits := inAtomDigits.set! (spec.inPos[j]!) (rem % s)
          rem := rem / s
        else
          inAtomDigits := inAtomDigits.set! (spec.inPos[j]!) 0

    let mut inIdx := Array.mkArray inDims.length 0
    for ax in [:inDims.length] do
      let d := inDims[ax]!
      let start := inAtomStarts[ax]!
      let atomCount := d.atoms.length
      let mut digs : Array Nat := Array.mkEmpty atomCount
      for j in [:atomCount] do
        digs := digs.push (inAtomDigits[start + j]!)
      let sizes := d.atoms.map (·.size) |>.toArray
      inIdx := inIdx.set! ax (axisIndexFromAtomDigits digs sizes)

    let flat := flatIndex inIdx t.strides t.offset
    resultData := resultData.push (t.data.get! flat)

  return { data := resultData
           shape := outShape
           strides := computeStrides outShape
           offset := 0 }

def Tensor.rearrange {inDims outDims : DimList} {α : Type} [Inhabited α] (t : Tensor inDims α)
    (_valid : validRearrange inDims outDims = true := by decide) : Tensor outDims α :=
  if hReshape : validReshape inDims outDims = true then
    let _ := hReshape
    let outShape := shapeOf outDims
    { data := t.data
      shape := outShape
      strides := computeStrides outShape
      offset := t.offset }
  else
    Tensor.rearrangeByRoots t

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

instance (A B : DimList) : EinsumOut PUnit A B where
  outDims := []
  outSrc := []

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

private def validEinsumByOutSrc (outSrc : List (Bool × Nat)) (aLen bLen : Nat) : Bool := Id.run do
  let mut aSeen : Array Bool := Array.mkArray aLen false
  let mut bSeen : Array Bool := Array.mkArray bLen false
  for (fromB, srcIdx) in outSrc do
    if fromB then
      if srcIdx < bLen then
        if bSeen[srcIdx]! then
          return false
        bSeen := bSeen.set! srcIdx true
      else
        return false
    else
      if srcIdx < aLen then
        if aSeen[srcIdx]! then
          return false
        aSeen := aSeen.set! srcIdx true
      else
        return false
  return true

private structure EinsumOperand (α : Type) where
  dims : DimList
  t : Tensor dims α

private instance {α : Type} [Inhabited α] : Inhabited (EinsumOperand α) where
  default := { dims := [], t := (default : Tensor [] α) }

private structure AxisPlan where
  isOut : Array Bool
  toOut : Array Nat
  toContr : Array Nat

instance : Inhabited AxisPlan where
  default := { isOut := #[], toOut := #[], toContr := #[] }

private def Tensor.einsumNCore {outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [Mul α]
    (ops : Array (EinsumOperand α))
    : Tensor outDims α := Id.run do
  let outShape := shapeOf outDims
  if ops.size == 0 then
    return { data := Array.mkArray (totalSize outShape) 0
             shape := outShape
             strides := computeStrides outShape
             offset := 0 }
  let outLabels := outDims.toArray

  -- Dims are labels: same dim = same index (trace/diagonal semantics for repeated dims).
  -- Collect unique dims across all operands, separate into output vs contraction.
  let allLabels := Id.run do
    let mut out : Array Dim := #[]
    for op in ops do
      for d in op.dims do
        if !(out.any (· == d)) then
          out := out.push d
    return out
  let mut reducedLabels : Array Dim := #[]
  for d in allLabels do
    if !(outLabels.any (· == d)) then
      reducedLabels := reducedLabels.push d
  let contrShape := reducedLabels.map (·.size)

  let plans := ops.map fun op => Id.run do
    let mut isOut := Array.mkArray op.dims.length false
    let mut toOut := Array.mkArray op.dims.length 0
    let mut toContr := Array.mkArray op.dims.length 0
    for p in [:op.dims.length] do
      let d := op.dims[p]!
      match findDimIndex? outLabels d with
      | some idx =>
          isOut := isOut.set! p true
          toOut := toOut.set! p idx
      | none =>
          match findDimIndex? reducedLabels d with
          | some cIdx => toContr := toContr.set! p cIdx
          | none => panic! "einsumN: dim not found in output or contraction labels"
    return { isOut, toOut, toContr : AxisPlan }

  let outTotal := totalSize outShape
  let contrTotal := totalSize contrShape
  let mut resultData : Array α := Array.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    let mut acc : α := 0
    for contrFlat in [:contrTotal] do
      let contrIdx := toMultiIndex contrFlat contrShape
      let firstOp := ops[0]!
      let firstPlan := plans[0]!
      let mut firstIdx := Array.mkArray firstOp.dims.length 0
      for p in [:firstOp.dims.length] do
        if firstPlan.isOut[p]! then
          firstIdx := firstIdx.set! p (outIdx[firstPlan.toOut[p]!]!)
        else
          firstIdx := firstIdx.set! p (contrIdx[firstPlan.toContr[p]!]!)
      let mut term := firstOp.t.get! firstIdx
      for opIdx in [1:ops.size] do
        let op := ops[opIdx]!
        let plan := plans[opIdx]!
        let mut idx := Array.mkArray op.dims.length 0
        for p in [:op.dims.length] do
          if plan.isOut[p]! then
            idx := idx.set! p (outIdx[plan.toOut[p]!]!)
          else
            idx := idx.set! p (contrIdx[plan.toContr[p]!]!)
        term := term * op.t.get! idx
      acc := acc + term
    resultData := resultData.push acc
  return { data := resultData
           shape := outShape
           strides := computeStrides outShape
           offset := 0 }

private def Tensor.einsum2Core {A B outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [Mul α]
    (x : Tensor A α) (y : Tensor B α)
    : Tensor outDims α :=
  Tensor.einsumNCore (outDims := outDims)
    #[{ dims := A, t := x }, { dims := B, t := y }]

private def Tensor.einsum2BySrc {A B outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [Mul α]
    (x : Tensor A α) (y : Tensor B α)
    (outSrc : Array (Bool × Nat))
    : Tensor outDims α := Id.run do
  let outShape := shapeOf outDims
  let outTotal := totalSize outShape
  -- Build output → operand axis mapping
  let mut aOutMap : Array (Option Nat) := Array.mkArray A.length none
  let mut bOutMap : Array (Option Nat) := Array.mkArray B.length none
  for oAxis in [:outSrc.size] do
    let (fromB, srcIdx) := outSrc[oAxis]!
    if fromB then bOutMap := bOutMap.set! srcIdx (some oAxis)
    else aOutMap := aOutMap.set! srcIdx (some oAxis)
  -- Contracted axes
  let mut aContr : Array Nat := #[]
  for p in [:A.length] do
    if (aOutMap[p]!).isNone then aContr := aContr.push p
  let mut bContr : Array Nat := #[]
  for p in [:B.length] do
    if (bOutMap[p]!).isNone then bContr := bContr.push p
  -- Pair contracted axes by dim value, append unpaired
  let mut contrSizes : Array Nat := #[]
  let mut contrAAxis : Array (Option Nat) := #[]
  let mut contrBAxis : Array (Option Nat) := #[]
  let mut bContrUsed := Array.mkArray bContr.size false
  for ai in aContr do
    let mut matched := false
    for bi_idx in [:bContr.size] do
      if !bContrUsed[bi_idx]! && !matched then
        let bi := bContr[bi_idx]!
        if A.get! ai == B.get! bi then
          contrSizes := contrSizes.push x.shape[ai]!
          contrAAxis := contrAAxis.push (some ai)
          contrBAxis := contrBAxis.push (some bi)
          bContrUsed := bContrUsed.set! bi_idx true
          matched := true
    if !matched then
      contrSizes := contrSizes.push x.shape[ai]!
      contrAAxis := contrAAxis.push (some ai)
      contrBAxis := contrBAxis.push none
  for bi_idx in [:bContr.size] do
    if !bContrUsed[bi_idx]! then
      let bi := bContr[bi_idx]!
      contrSizes := contrSizes.push y.shape[bi]!
      contrAAxis := contrAAxis.push none
      contrBAxis := contrBAxis.push (some bi)
  let contrTotal := totalSize contrSizes
  let mut resultData : Array α := Array.mkEmpty outTotal
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    let mut acc : α := 0
    for contrFlat in [:contrTotal] do
      let contrIdx := toMultiIndex contrFlat contrSizes
      let mut aIdx := Array.mkArray A.length 0
      for p in [:A.length] do
        match aOutMap[p]! with
        | some oAxis => aIdx := aIdx.set! p (outIdx[oAxis]!)
        | none => pure ()
      for ci in [:contrAAxis.size] do
        match contrAAxis[ci]! with
        | some ai => aIdx := aIdx.set! ai (contrIdx[ci]!)
        | none => pure ()
      let mut bIdx := Array.mkArray B.length 0
      for p in [:B.length] do
        match bOutMap[p]! with
        | some oAxis => bIdx := bIdx.set! p (outIdx[oAxis]!)
        | none => pure ()
      for ci in [:contrBAxis.size] do
        match contrBAxis[ci]! with
        | some bi => bIdx := bIdx.set! bi (contrIdx[ci]!)
        | none => pure ()
      acc := acc + x.get! aIdx * y.get! bIdx
    resultData := resultData.push acc
  return { data := resultData
           shape := outShape
           strides := computeStrides outShape
           offset := 0 }

def Tensor.einsumBy {A B : DimList} {Out : Type} {α : Type}
    [h : EinsumOut Out A B]
    [Inhabited α] [Zero α] [Add α] [Mul α]
    (x : Tensor A α) (y : Tensor B α)
    (_f : SlotTuple A → SlotTuple B → Out)
    (_validOut : validEinsumByOutSrc h.outSrc A.length B.length = true := by decide)
    : Tensor h.outDims α :=
  Tensor.einsum2BySrc (outDims := h.outDims) x y h.outSrc.toArray

class EinsumArgs (Ops : Type) (outDims : DimList) (α : Type) where
  validDims : Bool
  run : Ops → Tensor outDims α

instance {A B outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [Mul α] :
    EinsumArgs (Tensor A α × Tensor B α) outDims α where
  validDims := validEinsum2 A B outDims
  run := fun (x, y) => Tensor.einsum2Core (outDims := outDims) x y

instance {A B C outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [Mul α] :
    EinsumArgs (Tensor A α × Tensor B α × Tensor C α) outDims α where
  validDims := validEinsum3 A B C outDims
  run := fun (x, y, z) =>
    Tensor.einsumNCore (outDims := outDims)
      #[{ dims := A, t := x }, { dims := B, t := y }, { dims := C, t := z }]

def Tensor.einsum {Ops : Type} {outDims : DimList} {α : Type}
    [h : EinsumArgs Ops outDims α]
    (ops : Ops)
    (_valid : h.validDims = true := by decide) : Tensor outDims α :=
  h.run ops

-- ============================================
-- REDUCE
-- ============================================

private def Tensor.reduceGeneric {inDims outDims : DimList} {α : Type} [Inhabited α]
    (t : Tensor inDims α)
    (mapping : Array (Array Nat))
    (reducedAxes : Array Nat)
    (init : α) (combine : α → α → α) (finalize : α → Nat → α)
    : Tensor outDims α := Id.run do
  let outShape := shapeOf outDims
  let outTotal := totalSize outShape
  let reducedShape := reducedAxes.map (fun p => t.shape[p]!)
  let reducedTotal := totalSize reducedShape
  let mut resultData : Array α := Array.mkEmpty outTotal
  if reducedTotal == 0 then
    for _ in [:outTotal] do
      resultData := resultData.push init
    return { data := resultData, shape := outShape,
             strides := computeStrides outShape, offset := 0 }
  for outFlat in [:outTotal] do
    let outIdx := toMultiIndex outFlat outShape
    -- First reduced element to initialize accumulator
    let mut inIdx := Array.mkArray inDims.length 0
    -- Fill output-mapped axes
    for oAxis in [:mapping.size] do
      let axes := mapping[oAxis]!
      let mut remainder := outIdx[oAxis]!
      for rev in [:axes.size] do
        let aIdx := axes.size - 1 - rev
        let p := axes[aIdx]!
        let sz := t.shape[p]!
        inIdx := inIdx.set! p (remainder % sz)
        remainder := remainder / sz
    -- Fill reduced axes with first element (all zeros already)
    let flat0 := flatIndex inIdx t.strides t.offset
    let mut acc := t.data.get! flat0
    -- Iterate remaining reduced elements
    for redFlat in [1:reducedTotal] do
      let redIdx := toMultiIndex redFlat reducedShape
      let mut inIdx2 := inIdx
      for rAxis in [:reducedAxes.size] do
        inIdx2 := inIdx2.set! (reducedAxes[rAxis]!) (redIdx[rAxis]!)
      let flat := flatIndex inIdx2 t.strides t.offset
      let val := t.data.get! flat
      acc := combine acc val
    acc := finalize acc reducedTotal
    resultData := resultData.push acc
  return { data := resultData, shape := outShape,
           strides := computeStrides outShape, offset := 0 }

private def natToAlpha {α : Type} [Zero α] [Add α] [OfNat α 1] (n : Nat) : α := Id.run do
  let mut acc : α := 0
  for _ in [:n] do
    acc := acc + (OfNat.ofNat 1 : α)
  return acc

private def dispatchReduce {inDims outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [HDiv α α α] [Ord α] [OfNat α 1]
    (t : Tensor inDims α) (op : ReduceOp)
    (mapping : Array (Array Nat)) (reducedAxes : Array Nat)
    : Tensor outDims α :=
  match op with
  | .sum => Tensor.reduceGeneric t mapping reducedAxes 0
      (· + ·) (fun a _ => a)
  | .mean => Tensor.reduceGeneric t mapping reducedAxes 0
      (· + ·) (fun a n => a / natToAlpha n)
  | .max => Tensor.reduceGeneric t mapping reducedAxes 0
      (fun a b => if compare a b == .lt then b else a) (fun a _ => a)
  | .min => Tensor.reduceGeneric t mapping reducedAxes 0
      (fun a b => if compare a b == .gt then b else a) (fun a _ => a)

def Tensor.reduceBy {dims : DimList} {Out : Type}
    [h : HasDims Out dims]
    {α : Type} [Inhabited α] [Zero α] [Add α] [HDiv α α α] [Ord α] [OfNat α 1]
    (t : Tensor dims α) (op : ReduceOp) (_f : SlotTuple dims → Out)
    : Tensor h.outDims α := Id.run do
  let mapping := h.outMapping.map (·.toArray) |>.toArray
  let mut usedAxes : Array Bool := Array.mkArray dims.length false
  for oIdx in [:mapping.size] do
    let axes := mapping[oIdx]!
    for aIdx in [:axes.size] do
      usedAxes := usedAxes.set! (axes[aIdx]!) true
  let mut reducedAxes : Array Nat := #[]
  for p in [:dims.length] do
    if !(usedAxes[p]!) then
      reducedAxes := reducedAxes.push p
  return dispatchReduce t op mapping reducedAxes

def Tensor.reduce {inDims outDims : DimList} {α : Type}
    [Inhabited α] [Zero α] [Add α] [HDiv α α α] [Ord α] [OfNat α 1]
    (t : Tensor inDims α) (op : ReduceOp)
    (_valid : validReduce inDims outDims = true := by decide)
    : Tensor outDims α :=
  let (mapping, reducedAxes) := computeReduceInfo inDims outDims
  dispatchReduce t op mapping reducedAxes

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

def cmat : Tensor [i, j] :=
  Tensor.einsumBy a bmat (fun (i, _) (_, j) => (i, j))

def cmat2 : Tensor [i, j] := Tensor.einsum (a, bmat)

-- Compile-time validation test: uncomment to see error
-- def bogus := dim! 99
-- def badEinsum : Tensor [bogus] Nat := Tensor.einsum (a, bmat)

-- Unary counterparts via rearrange/reduce
def smallTU : Tensor [dj, di] := small.rearrange
def rowSumsU : Tensor [di] := small.reduceBy .sum fun (i, _) => i
def colSumsU : Tensor [dj] := small.reduceBy .sum fun (_, j) => j
def totalSumU : Tensor [] := small.reduceBy .sum fun (_, _) => ()

-- Scalar-output dot products and Hadamard product
def dotV1 : Tensor [dj] := arange 1
def dotV2 : Tensor [dj] := arange 3
def vecDot : Tensor [] := Tensor.einsum (dotV1, dotV2)

def dotM1 : Tensor [di, dj] := arange 1
def dotM2 : Tensor [di, dj] := arange 10
def matDot : Tensor [] := Tensor.einsum (dotM1, dotM2)
def hadamard : Tensor [di, dj] := Tensor.einsum (dotM1, dotM2)

-- Batch matrix multiplication: bik,bkj->bij
def eb := dim! 2
def ba : Tensor [eb, i, k] := arange 1
def bbatch : Tensor [eb, k, j] := arange 1
def bmm : Tensor [eb, i, j] := Tensor.einsum (ba, bbatch)

-- 3-input bilinear: ik,jkl,il->ij
def el := dim! 2
def bilX : Tensor [i, k] := arange 1
def bilW : Tensor [j, k, el] := arange 1
def bilY : Tensor [i, el] := arange 1
def bilinear : Tensor [i, j] := Tensor.einsum (bilX, bilW, bilY)

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

-- Einops-style flatten: "b c h w -> b (c h w)"
def chw := c * h * w
def flattenedCHW : Tensor [b, chw] := image.rearrange
-- Keep this as a type-level example only (large runtime tensor).
-- Use the small variant below for #eval.

-- Small flatten demo to keep #eval fast
def sb := dim! 2
def sc := dim! 3
def sh := dim! 2
def sw := dim! 2
def small4d : Tensor [sb, sc, sh, sw] := arange 1
def schws := sc * sh * sw
def flattenedSmallCHW : Tensor [sb, schws] := small4d.rearrange

-- Small merge test: 2×3 -> 6 (flatten)
#eval small      -- [[1, 2, 3], [4, 5, 6]]
#eval smallT     -- [[1, 4], [2, 5], [3, 6]]
#eval cmat       -- [[92, 98, 104, 110], [218, 233, 248, 263]]
#eval cmat2      -- [[92, 98, 104, 110], [218, 233, 248, 263]]
#eval smallTU    -- [[1, 4], [2, 5], [3, 6]]
#eval rowSumsU   -- [6, 15]
#eval colSumsU   -- [5, 7, 9]
#eval totalSumU  -- 21
#eval vecDot     -- 26
#eval matDot     -- 280
#eval hadamard   -- [[10, 22, 36], [52, 70, 90]]
#eval bmm
#eval bilinear
#eval flattenedSmallCHW.shape -- #[2, 12]
#eval flattenedSmallCHW

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

namespace batch
-- batch transformations
def b := dim! 6
def h := dim! 2
def w := dim! 2

factor! b1, b2 := b, 2

def example2d : Tensor [b, h, w] := arange 1
def rep : Tensor [b1 * h, b2 * w] := example2d.rearrange
def rep2 : Tensor [b2 * w, b1 * h] := example2d.rearrange
def rep3 : Tensor [b2 * h, b1 * w] := example2d.rearrange

#eval example2d
#eval rep
#eval rep2
#eval rep3

end batch

-- ============================================
-- REDUCE EXAMPLES
-- ============================================

-- small = [[1,2,3],[4,5,6]], shape [di=2, dj=3]

-- Sum over columns: [2,3] -> [2]
-- Expected: [6, 15]  (1+2+3=6, 4+5+6=15)
def rowSums : Tensor [di] := small.reduceBy .sum fun (i, _) => i
#eval rowSums

-- Sum over rows: [2,3] -> [3]
-- Expected: [5, 7, 9]  (1+4=5, 2+5=7, 3+6=9)
def colSums : Tensor [dj] := small.reduceBy .sum fun (_, j) => j
#eval colSums

-- Mean over columns: [2,3] -> [2]
-- Expected: [2, 5]  (6/3=2, 15/3=5)
def rowMeans : Tensor [di] := small.reduceBy .mean fun (i, _) => i
#eval rowMeans

-- Max over columns: [2,3] -> [2]
-- Expected: [3, 6]
def rowMax : Tensor [di] := small.reduceBy .max fun (i, _) => i
#eval rowMax

-- Min over rows: [2,3] -> [3]
-- Expected: [1, 2, 3]
def colMin : Tensor [dj] := small.reduceBy .min fun (_, j) => j
#eval colMin

-- Type-driven reduce (no lambda): [2,3] -> [2]
def rowSums2 : Tensor [di] := small.reduce .sum
#eval rowSums2

-- Full reduction to scalar: [2,3] -> []
-- Expected: [21]  (sum of all elements)
def totalSum : Tensor [] := small.reduceBy .sum fun (_, _) => ()
#eval totalSum

end Einlean
