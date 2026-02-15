import Einlean.Core

namespace Einlean2.Test
open Einlean2

private def checkShape (name : String) (actual expected : Array Nat) : IO Unit :=
  if actual == expected then
    IO.println s!"  PASS: {name}"
  else
    throw (IO.userError s!"  FAIL: {name}\n    expected: {expected}\n    actual:   {actual}")

private def checkNat (name : String) (actual expected : Nat) : IO Unit :=
  if actual == expected then
    IO.println s!"  PASS: {name}"
  else
    throw (IO.userError s!"  FAIL: {name}\n    expected: {expected}\n    actual:   {actual}")

abbrev di := dim! 2
abbrev dj := dim! 3
abbrev dk := dim! 4

def small : Tensor [di, dj] := Tensor.init
def vec : Tensor [dj] := Tensor.init

#eval do
  IO.println "=== Core: Shape-only Creation ==="
  checkShape "2x3 shape" small.shape #[2, 3]
  checkShape "vec shape" vec.shape #[3]

-- ============================================
-- REARRANGE
-- ============================================

def smallT : Tensor [dj, di] :=
  small.rearrangeBy fun (i, (j, ())) => (j, (i, ()))

def smallId : Tensor [di, dj] :=
  small.rearrangeBy fun idx => idx

#eval do
  IO.println "=== Core: Rearrange ==="
  checkShape "transpose shape" smallT.shape #[3, 2]
  checkShape "identity shape" smallId.shape #[2, 3]

-- ============================================
-- RESHAPE
-- ============================================

def flat : Tensor [di * dj] := Tensor.init
def unflat : Tensor [di, dj] := flat.reshape

#eval do
  IO.println "=== Core: Reshape ==="
  checkShape "flat shape" flat.shape #[6]
  checkShape "unflat shape" unflat.shape #[2, 3]

-- ============================================
-- REDUCE
-- ============================================

def rowSums : Tensor [di] :=
  small.reduceBy (.sum) fun (i, _) => i

def colSums : Tensor [dj] :=
  small.reduceBy (.sum) fun (_, j) => j

def totalSum : Tensor (α := Int) [] :=
  small.reduceBy (.sum) fun _ => scalar

#eval do
  IO.println "=== Core: Reduce ==="
  checkShape "row sums shape" rowSums.shape #[2]
  checkShape "col sums shape" colSums.shape #[3]
  checkShape "total sum shape" totalSum.shape #[]

-- ============================================
-- EINSUM (matmul simulation)
-- ============================================

def mat34 : Tensor [dj, dk] := Tensor.init

set_option linter.unusedVariables false in
def matmul : Tensor [di, dk] :=
  Tensor.einsumBy small mat34
    (fun (i, _k) (_k2, j) => (i, j))

def v2 : Tensor [dj] := Tensor.init

set_option linter.unusedVariables false in
def vdot : Tensor (α := Int) [] :=
  Tensor.einsumBy vec v2
    (fun _ _ => scalar)

#eval do
  IO.println "=== Core: Einsum ==="
  checkShape "matmul shape" matmul.shape #[2, 4]
  checkShape "dot product shape" vdot.shape #[]

-- ============================================
-- 3D REARRANGE
-- ============================================

abbrev db := dim! 2
abbrev dh := dim! 3
abbrev dw := dim! 4

def img : Tensor [db, dh, dw] := Tensor.init

def imgRearranged : Tensor [dh, dw, db] :=
  img.rearrangeBy fun (b, (h, (w, ()))) => (h, (w, (b, ())))

#eval do
  IO.println "=== Core: 3D Rearrange ==="
  checkShape "rearranged shape" imgRearranged.shape #[3, 4, 2]

-- ============================================
-- REPEATED DIM
-- ============================================

abbrev rd := dim! 2

def sqMat : Tensor [rd, rd] := Tensor.init

def sqT : Tensor [rd, rd] :=
  sqMat.rearrangeBy fun (i, (j, ())) => (j, (i, ()))

def sqRowSums : Tensor [rd] :=
  sqMat.reduceBy (.sum) fun (i, _) => i

set_option linter.unusedVariables false in
def sqColSums : Tensor [rd] :=
  sqMat.reduceBy (.sum) fun (_, j) => j

#eval do
  IO.println "=== Core: Repeated Dim ==="
  checkShape "2x2 shape" sqMat.shape #[2, 2]
  checkShape "2x2 transpose shape" sqT.shape #[2, 2]
  checkShape "2x2 row sums shape" sqRowSums.shape #[2]
  checkShape "2x2 col sums shape" sqColSums.shape #[2]

abbrev ri := dim! 2
abbrev rj := dim! 3

def sqMat2 : Tensor [ri, ri] := Tensor.init
def rect : Tensor [ri, rj] := Tensor.init

def sqTimesRect : Tensor [ri, rj] :=
  Tensor.einsumBy sqMat2 rect (fun (i, _) (_, j) => (i, j))

#eval do
  IO.println "=== Core: Einsum Ergonomics ==="
  checkShape "sqTimesRect shape" sqTimesRect.shape #[2, 3]

-- ============================================
-- BATCH MATMUL
-- ============================================

abbrev db2 := dim! 2
abbrev di2 := dim! 2
abbrev dj2 := dim! 3
abbrev dk2 := dim! 4

def batchA : Tensor [db2, di2, dj2] := Tensor.init
def batchB : Tensor [db2, dj2, dk2] := Tensor.init

set_option linter.unusedVariables false in
def batchMM : Tensor [db2, di2, dk2] :=
  Tensor.einsumBy batchA batchB
    (fun (b, i, _k) (_b2, _k2, j) => (b, i, j))

#eval do
  IO.println "=== Core: Batch Matmul ==="
  checkShape "batch mm shape" batchMM.shape #[2, 2, 4]

-- ============================================
-- DECOMPOSITION / FACTOR!
-- ============================================

namespace Decomp

abbrev b := dim! 6
abbrev h := dim! 3
abbrev w := dim! 4
abbrev c := dim! 2

factor! b1, b2 := b, 2
def _witness : PackOf (b1 * b2) = (Fin b1.size × Fin b2.size) := by rfl

def bPair := (b1, b2)
def imgFromPair : Tensor [bPair, h, w] := Tensor.init
def imgFromPair2 : Tensor [(b1, b2), h, w] := imgFromPair

#eval do
  IO.println "=== Core: Decomposition ==="
  checkNat "b1 size" b1.size 3
  checkNat "b2 size" b2.size 2

#check Einlean2.instDecompMul

#check DecompDim
#check Einlean2.DecompDim

set_option trace.Meta.synthInstance true in
#synth Einlean2.DecompDim (b1 * b2)

#check (inferInstance : DecompDim (b1 * b2))
#check instDecompMul

def img : Tensor [b, h, w] := Tensor.init
def img2 : Tensor [(b1, b2), h, w] := img
def img2e : Tensor [b1 * b2, h, w] := img2

section
  local instance : DecompDim (b1 * b2) := decompMul b1 b2

  def split : Tensor [b1, b2, h, w] :=
    img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
      (b1i, (b2i, (hi, (wi, ()))))
end



def mergedPair : Tensor [b1 * b2, h, w] :=
  split.rearrangeBy fun (b1i, (b2i, (hi, (wi, ())))) =>
    ((b1i, b2i), (hi, (wi, ())))

def merged : Tensor [b, h, w] := mergedPair

#eval do
  checkShape "split shape" split.shape #[3, 2, 3, 4]
  checkShape "merged shape" merged.shape #[6, 3, 4]

def fancy : Tensor [h, b1, w, b2] :=
  img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
    (hi, (b1i, (wi, (b2i, ()))))

#eval do
  checkShape "fancy shape" fancy.shape #[3, 3, 4, 2]

def flatImgPair : Tensor [b1 * b2, h * w] :=
  img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
    ((b1i, b2i), ((hi, wi), ()))

def flatImg : Tensor [b, h * w] := flatImgPair

def unflatImgPair : Tensor [b1 * b2, h, w] :=
  flatImgPair.rearrangeBy fun ((b1i, b2i), ((hi, wi), ())) =>
    ((b1i, b2i), (hi, (wi, ())))

def unflatImg : Tensor [b, h, w] := unflatImgPair

#eval do
  checkShape "flat shape" flatImg.shape #[6, 12]
  checkShape "unflat shape" unflatImg.shape #[6, 3, 4]

-- ============================================
-- 2x2 MEAN-POOLING SHAPE PIPELINE
-- ============================================

abbrev pb := dim! 2
abbrev ph := dim! 4
abbrev pw := dim! 4
abbrev pc := dim! 1

factor! phOut, ph2 := ph, 2
factor! pwOut, pw2 := pw, 2

def poolInput : Tensor [pb, ph, pw, pc] := Tensor.init
def poolInput2 : Tensor [pb, phOut * ph2, pwOut * pw2, pc] := poolInput

def expanded : Tensor [pb, phOut, ph2, pwOut, pw2, pc] :=
  poolInput2.rearrangeBy fun (pbi, ((phOi, ph2i), ((pwOi, pw2i), (pci, ())))) =>
    (pbi, (phOi, (ph2i, (pwOi, (pw2i, (pci, ()))))))

set_option linter.unusedVariables false in
def pooled : Tensor [phOut, pb, pwOut, pc] :=
  expanded.reduceBy (.sum) fun (pbi, phOi, _ph2i, pwOi, _pw2i, pci) =>
    (phOi, pbi, pwOi, pci)

#eval do
  IO.println "=== Core: Pooling Shape Pipeline ==="
  checkShape "expanded shape" expanded.shape #[2, 2, 2, 2, 2, 1]
  checkShape "pooled shape" pooled.shape #[2, 2, 2, 1]

end Decomp

end Einlean2.Test
