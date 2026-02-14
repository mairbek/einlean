import Einlean

namespace EinleanTest
open Einlean

private def check (name : String) (actual expected : List Int) : IO Unit :=
  if actual == expected then
    IO.println s!"  PASS: {name}"
  else
    throw (IO.userError s!"  FAIL: {name}\n    expected: {expected}\n    actual:   {actual}")

private def checkShape (name : String) (actual expected : Array Nat) : IO Unit :=
  if actual == expected then
    IO.println s!"  PASS: {name}"
  else
    throw (IO.userError s!"  FAIL: {name}\n    expected: {expected}\n    actual:   {actual}")

-- ============================================
-- REPEATED DIM: CREATION
-- ============================================

def ri := dim! 2

def sqMat : Tensor [ri, ri] := arange 1

#eval do
  IO.println "=== Repeated Dim: Creation ==="
  check "2x2 arange" sqMat.toList [1, 2, 3, 4]
  checkShape "2x2 shape" sqMat.shape #[2, 2]

-- ============================================
-- REPEATED DIM: REARRANGE (lambda-based)
-- ============================================

def sqMatT : Tensor [ri, ri] :=
  sqMat.rearrangeBy fun (a, b) => (b, a)

def sqMatId : Tensor [ri, ri] :=
  sqMat.rearrangeBy fun (a, b) => (a, b)

#eval do
  IO.println "=== Repeated Dim: Rearrange ==="
  check "transpose" sqMatT.toList [1, 3, 2, 4]
  check "identity" sqMatId.toList [1, 2, 3, 4]

-- ============================================
-- REPEATED DIM: REDUCE (lambda-based)
-- ============================================

-- [[1,2],[3,4]]
-- Row sums: [1+2, 3+4] = [3, 7]
def sqRowSums : Tensor [ri] :=
  sqMat.reduceBy .sum fun (a, _) => a

-- Col sums: [1+3, 2+4] = [4, 6]
set_option linter.unusedVariables false in
def sqColSums : Tensor [ri] :=
  sqMat.reduceBy .sum fun (_, b) => b

-- Total sum: 1+2+3+4 = 10
set_option linter.unusedVariables false in
def sqTotalSum : Tensor [] :=
  sqMat.reduceBy .sum fun (_, _) => ()

-- Row max: [max(1,2), max(3,4)] = [2, 4]
def sqRowMax : Tensor [ri] :=
  sqMat.reduceBy .max fun (a, _) => a

-- Row mean: [(1+2)/2, (3+4)/2] = [1, 3] (integer division)
def sqRowMean : Tensor [ri] :=
  sqMat.reduceBy .mean fun (a, _) => a

#eval do
  IO.println "=== Repeated Dim: Reduce ==="
  check "row sums" sqRowSums.toList [3, 7]
  check "col sums" sqColSums.toList [4, 6]
  check "total sum" sqTotalSum.toList [10]
  check "row max" sqRowMax.toList [2, 4]
  check "row mean" sqRowMean.toList [1, 3]

-- ============================================
-- REPEATED DIM: EINSUM (one operand has repeated dim)
-- ============================================

def rj := dim! 3

-- sqMat = [[1,2],[3,4]] (2x2), rect = [[1,2,3],[4,5,6]] (2x3)
def rect : Tensor [ri, rj] := arange 1

-- sqMat × rect = [[1*1+2*4, 1*2+2*5, 1*3+2*6],
--                  [3*1+4*4, 3*2+4*5, 3*3+4*6]]
--             = [[9, 12, 15], [19, 26, 33]]
set_option linter.unusedVariables false in
def sqTimesRect : Tensor [ri, rj] :=
  Tensor.einsumBy sqMat rect (fun (_i, _k) (_k2, _j) => (_i, _j))

-- rect^T × sqMat = (3x2) × (2x2)
-- rect^T = [[1,4],[2,5],[3,6]]
-- rect^T × sqMat = [[1*1+4*3, 1*2+4*4], [2*1+5*3, 2*2+5*4], [3*1+6*3, 3*2+6*4]]
--                = [[13, 18], [17, 24], [21, 30]]
def rectT : Tensor [rj, ri] :=
  rect.rearrangeBy fun (a, b) => (b, a)

set_option linter.unusedVariables false in
def rectTTimesSq : Tensor [rj, ri] :=
  Tensor.einsumBy rectT sqMat (fun (_j, _k) (_k2, _i) => (_j, _i))

-- Tuple API with repeated dims: same dim = same index (diagonal semantics).
-- sqMat = [[1,2],[3,4]], rect = [[1,2,3],[4,5,6]]
-- ii,ij->ij: out[i,j] = diag(sqMat)[i] * rect[i,j]
-- = [[1*1, 1*2, 1*3], [4*4, 4*5, 4*6]] = [[1,2,3],[16,20,24]]
def sqTimesRectTuple : Tensor [ri, rj] := Tensor.einsum (sqMat, rect)

-- IMPORTANT SAFETY DEMO:
-- einsumBy now rejects duplicate output-axis selection at compile time.
-- Previously, duplicates could silently overwrite mapping and produce surprising results.
-- Uncommenting the line below should fail elaboration:
-- def badDupBy : Tensor [ri, ri] :=
--   Tensor.einsumBy sqMat rect (fun (i, _) (_, _j) => (i, i))
-- Use tuple einsum when you want repeated-dim/diagonal semantics.

-- ji,ii->ji: out[j,i] = rectT[j,i] * diag(sqMat)[i]
-- rectT = [[1,4],[2,5],[3,6]], diag = [1,4]
-- = [[1*1, 4*4], [2*1, 5*4], [3*1, 6*4]] = [[1,16],[2,20],[3,24]]
def rectTimesSqTuple : Tensor [rj, ri] := Tensor.einsum (rectT, sqMat)

#eval do
  IO.println "=== Repeated Dim: Einsum ==="
  -- einsumBy: user explicitly selects which axes are contracted (matmul)
  check "sqMat × rect (einsumBy)" sqTimesRect.toList [9, 12, 15, 19, 26, 33]
  check "rectT × sqMat (einsumBy)" rectTTimesSq.toList [13, 18, 17, 24, 21, 30]
  -- tuple API: same dim = same index, so repeated dims give diagonal behavior
  check "sqMat diag × rect (tuple)" sqTimesRectTuple.toList [1, 2, 3, 16, 20, 24]
  check "rectT × sqMat diag (tuple)" rectTimesSqTuple.toList [1, 16, 2, 20, 3, 24]

-- ============================================
-- REGRESSION: DISTINCT DIMS
-- ============================================

namespace Regression

def di := dim! 2
def dj := dim! 3
def dk := dim! 4

-- 2×3 matrix [[1,2,3],[4,5,6]]
def mat23 : Tensor [di, dj] := arange 1
-- 3×4 matrix [[10,11,12,13],[14,15,16,17],[18,19,20,21]]
def mat34 : Tensor [dj, dk] := arange 10

-- Transpose (lambda-based)
def mat23T : Tensor [dj, di] :=
  mat23.rearrangeBy fun (i, j) => (j, i)

-- Transpose (atom-based)
def mat23T2 : Tensor [dj, di] := mat23.rearrange

-- Matmul (lambda-based)
set_option linter.unusedVariables false in
def matmul : Tensor [di, dk] :=
  Tensor.einsumBy mat23 mat34 (fun (i, _k) (_k2, j) => (i, j))

-- Matmul (atom-based)
def matmul2 : Tensor [di, dk] := Tensor.einsum (mat23, mat34)

-- Row sums (lambda-based)
def rowSums : Tensor [di] :=
  mat23.reduceBy .sum fun (i, _) => i

-- Row sums (atom-based)
def rowSums2 : Tensor [di] := mat23.reduce .sum

-- Col sums
set_option linter.unusedVariables false in
def colSums : Tensor [dj] :=
  mat23.reduceBy .sum fun (_, j) => j

-- Total sum
set_option linter.unusedVariables false in
def totalSum : Tensor [] :=
  mat23.reduceBy .sum fun (_, _) => ()

-- Reshape
def flat : Tensor [di * dj] := arange
def reshaped : Tensor [di, dj] := flat.reshape

-- Vector dot product
def v1 : Tensor [dj] := arange 1
def v2 : Tensor [dj] := arange 3
def vdot : Tensor [] := Tensor.einsum (v1, v2)

-- Batch matmul
def db := dim! 2
def batchA : Tensor [db, di, dj] := arange 1
def batchB : Tensor [db, dj, dk] := arange 1
def batchMM : Tensor [db, di, dk] := Tensor.einsum (batchA, batchB)

-- Merge dims
def merged : Tensor [di * dj] :=
  mat23.rearrangeBy fun (i, j) => i * j

#eval do
  IO.println "=== Regression: Distinct Dims ==="
  -- Transpose
  check "transpose (lambda)" mat23T.toList [1, 4, 2, 5, 3, 6]
  check "transpose (atom)" mat23T2.toList [1, 4, 2, 5, 3, 6]
  -- Matmul: [[1,2,3],[4,5,6]] × [[10,11,12,13],[14,15,16,17],[18,19,20,21]]
  -- Row 0: [1*10+2*14+3*18, 1*11+2*15+3*19, 1*12+2*16+3*20, 1*13+2*17+3*21]
  --       = [92, 98, 104, 110]
  -- Row 1: [4*10+5*14+6*18, ...] = [218, 233, 248, 263]
  check "matmul (lambda)" matmul.toList [92, 98, 104, 110, 218, 233, 248, 263]
  check "matmul (atom)" matmul2.toList [92, 98, 104, 110, 218, 233, 248, 263]
  -- Reduce
  check "row sums (lambda)" rowSums.toList [6, 15]
  check "row sums (atom)" rowSums2.toList [6, 15]
  check "col sums" colSums.toList [5, 7, 9]
  check "total sum" totalSum.toList [21]
  -- Reshape
  check "reshape" reshaped.toList [0, 1, 2, 3, 4, 5]
  -- Dot product: [1,2,3]·[3,4,5] = 3+8+15 = 26
  check "vector dot" vdot.toList [26]
  -- Merge
  check "merge dims" merged.toList [1, 2, 3, 4, 5, 6]

end Regression

-- ============================================
-- EINOPS-STYLE 2x2 MEAN POOLING
-- ============================================

namespace MeanPool

def pb := dim! 2
def ph := dim! 4
def pw := dim! 4
def pc := dim! 1

factor! phOut, ph2 := ph, 2
factor! pwOut, pw2 := pw, 2

-- Einops equivalent:
-- reduce(ims, "b (h h2) (w w2) c -> h (b w) c", "mean", h2=2, w2=2)
def pooled : Tensor [phOut, pb * pwOut, pc] := Id.run do
  let ims : Tensor [pb, ph, pw, pc] := arange 1
  let patches : Tensor [pb, phOut, ph2, pwOut, pw2, pc] := ims.rearrange
  return patches.reduceBy .mean fun (b, h, _h2, w, _w2, c) => (h, b * w, c)

#eval do
  IO.println "=== Mean Pool 2x2 ==="
  checkShape "pooled shape" pooled.shape #[2, 4, 1]
  -- h0: [3, 5, 19, 21], h1: [11, 13, 27, 29]
  check "pooled values" pooled.toList [3, 5, 19, 21, 11, 13, 27, 29]

end MeanPool

-- ============================================
-- 3x3 REPEATED DIM
-- ============================================

namespace ThreeByThree

def d := dim! 3

def mat : Tensor [d, d] := arange 1
-- [[1,2,3],[4,5,6],[7,8,9]]

def matT : Tensor [d, d] :=
  mat.rearrangeBy fun (a, b) => (b, a)

def rowSums : Tensor [d] :=
  mat.reduceBy .sum fun (a, _) => a

set_option linter.unusedVariables false in
def colSums : Tensor [d] :=
  mat.reduceBy .sum fun (_, b) => b

#eval do
  IO.println "=== 3x3 Repeated Dim ==="
  check "3x3 creation" mat.toList [1, 2, 3, 4, 5, 6, 7, 8, 9]
  check "3x3 transpose" matT.toList [1, 4, 7, 2, 5, 8, 3, 6, 9]
  check "3x3 row sums" rowSums.toList [6, 15, 24]
  check "3x3 col sums" colSums.toList [12, 15, 18]

end ThreeByThree

-- ============================================
-- SLICE / INDEXING WITH REPEATED DIMS
-- ============================================

def sqRow0 : Tensor [ri] := sqMat[0]
def sqRow1 : Tensor [ri] := sqMat[1]

#eval do
  IO.println "=== Repeated Dim: Indexing ==="
  check "row 0" sqRow0.toList [1, 2]
  check "row 1" sqRow1.toList [3, 4]

end EinleanTest
