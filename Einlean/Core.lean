/-
# Einlean Core

Minimal shape-only core for iterating quickly on the typesafe API.
Tensor values are phantom; only type-driven shapes are tracked.
-/

namespace Einlean2

-- ============================================
-- DIMENSIONS
-- ============================================

structure Dim where
  id   : Nat
  size : Nat
  hpos : 0 < size := by omega
  deriving Repr

instance : BEq Dim where
  beq a b := a.id == b.id && a.size == b.size

instance : DecidableEq Dim := fun a b =>
  if h1 : a.id = b.id then
    if h2 : a.size = b.size then
      isTrue (by cases a; cases b; simp_all)
    else isFalse (by intro h; cases h; exact h2 rfl)
  else isFalse (by intro h; cases h; exact h1 rfl)

instance : Inhabited Dim where
  default := { id := 0, size := 1 }

def dim (id : Nat) (size : Nat) (hpos : 0 < size := by omega) : Dim := { id, size, hpos }

open Lean in
scoped macro "dim!" size:term : term => do
  let some pos := (← getRef).getPos? | Macro.throwError "dim!: no source position"
  `(dim $(quote pos.byteIdx) $size)

def Dim.mul (a b : Dim) : Dim :=
  { id := a.id * 1000003 + b.id
  , size := a.size * b.size
  , hpos := Nat.mul_pos a.hpos b.hpos }

instance : Mul Dim := ⟨Dim.mul⟩

instance : Coe (Dim × Dim) Dim where
  coe p := p.1 * p.2

-- Canonical decomposition of a dim's flat index.
class DecompDim (d : Dim) where
  Pack : Type
  toLin   : Pack → Fin d.size
  fromLin : Fin d.size → Pack

private theorem pack_bound {a b i j : Nat} (hi : i < a) (hj : j < b) :
    i * b + j < a * b :=
  Nat.lt_of_lt_of_le
    (Nat.lt_of_lt_of_le (Nat.add_lt_add_left hj _)
      (Nat.le_of_eq (Nat.succ_mul i b).symm))
    (Nat.mul_le_mul_right b (Nat.succ_le_of_lt hi))

-- ============================================
-- DECOMPOSED DIMENSIONS
-- ============================================

theorem Dim.mul_size (a b : Dim) : (a * b).size = a.size * b.size := rfl


def decompMul (a b : Dim) : DecompDim (a * b) where
  Pack := Fin a.size × Fin b.size
  toLin := fun (i, j) =>
    ⟨i.val * b.size + j.val, pack_bound i.isLt j.isLt⟩
  fromLin := fun n =>
    have hn : n.val < a.size * b.size := n.isLt
    let i : Fin a.size := ⟨n.val / b.size, by
      have : n.val < b.size * a.size := by rw [Nat.mul_comm]; exact hn
      exact Nat.div_lt_of_lt_mul this⟩
    let j : Fin b.size := ⟨n.val % b.size, Nat.mod_lt _ b.hpos⟩
    (i, j)

-- Decompose a product dim as a pair.
instance (priority := 5) instDecompMul (a b : Dim) : DecompDim (a * b) :=
  decompMul a b

-- Fallback primitive decomposition: only if nothing more specific exists.
instance (priority := 10000) instDecompPrimitive (d : Dim) : DecompDim d where
  Pack := Fin d.size
  toLin := id
  fromLin := id


-- ============================================
-- FACTOR / DECOMPOSE
-- ============================================

def decompFactor (d : Dim) (outer inner : Dim)
    (h : d.size = outer.size * inner.size := by decide) : DecompDim d where
  Pack := Fin outer.size × Fin inner.size
  toLin := fun (i, j) =>
    ⟨i.val * inner.size + j.val, h ▸ pack_bound i.isLt j.isLt⟩
  fromLin := fun n =>
    have hn : n.val < outer.size * inner.size := h ▸ n.isLt
    (⟨n.val / inner.size, by
      have : n.val < inner.size * outer.size := by rw [Nat.mul_comm]; exact hn
      exact Nat.div_lt_of_lt_mul this⟩,
     ⟨n.val % inner.size, Nat.mod_lt _ inner.hpos⟩)

open Lean in
syntax "factor! " ident "," ident " := " term "," term : command

open Lean Macro in
macro "factor! " outer:ident "," inner:ident " := " d:term "," k:term : command => do
  `(abbrev $outer : Dim := { id := ($d).id, size := ($d).size / $k, hpos := by decide }
    abbrev $inner : Dim := { id := ($d).id + 1, size := $k, hpos := by decide }
    instance (priority := 5) : DecompDim $d := decompFactor $d $outer $inner
    )

abbrev PackOf (d : Dim) [DecompDim d] : Type := DecompDim.Pack (d := d)


def Dim.split (d : Dim) (i : Fin d.size) : PackOf d :=
  DecompDim.fromLin i

def Dim.merge (d : Dim) (x : PackOf d) : Fin d.size :=
  DecompDim.toLin x


-- ============================================
-- SHAPE / INDEX TYPES
-- ============================================

abbrev Shape := List Dim

def shapeSize : Shape → Nat
  | [] => 1
  | d :: ds => d.size * shapeSize ds

def shapeOf (ds : Shape) : List Nat :=
  ds.map (·.size)

abbrev Idx : Shape → Type
  | [] => Unit
  | [d] => Fin d.size
  | d :: ds => Fin d.size × Idx ds

def scalar : Idx [] := ()

abbrev PIdx : Shape → Type
  | []      => Unit
  | d :: ds => PackOf d × PIdx ds

-- ============================================
-- TENSOR (SHAPE-ONLY)
-- ============================================

structure TensorData (α : Type := Int) where
  shape : Array Nat

abbrev Tensor (_ds : Shape) (α : Type := Int) := TensorData α

instance {ds : Shape} {α : Type} : Inhabited (Tensor ds α) where
  default := { shape := (shapeOf ds).toArray }

def Tensor.init {ds : Shape} {α : Type} : Tensor ds α :=
  { shape := (shapeOf ds).toArray }

-- ============================================
-- REARRANGE / RESHAPE / REDUCE / EINSUM
-- (API surface only; shape-level simulation)
-- ============================================

def Tensor.rearrangeBy {ds es : Shape} {α : Type}
    (_t : Tensor ds α)
    (_fwd : PIdx ds → PIdx es) :
    Tensor es α :=
  Tensor.init

def Tensor.reshape {ds es : Shape} {α : Type}
    (_t : Tensor ds α)
    (_h : shapeSize ds = shapeSize es := by decide) :
    Tensor es α :=
  Tensor.init

structure Reducer (α : Type) where
  init : α
  step : α → α → α

namespace Reducer

def sum {α : Type} [Add α] [Zero α] : Reducer α := ⟨0, (· + ·)⟩

def max {α : Type} [Max α] (init : α) : Reducer α :=
  ⟨init, fun a b => Max.max a b⟩

def min {α : Type} [Min α] (init : α) : Reducer α :=
  ⟨init, fun a b => Min.min a b⟩

end Reducer

def Tensor.reduceBy {ds es : Shape} {α : Type}
    (_t : Tensor ds α)
    (_r : Reducer α)
    (_proj : Idx ds → Idx es) :
    Tensor es α :=
  Tensor.init

def Tensor.einsumBy {dsA dsB dsOut : Shape} {α : Type}
    (_a : Tensor dsA α) (_b : Tensor dsB α)
    (_out : Idx dsA → Idx dsB → Idx dsOut) :
    Tensor dsOut α :=
  Tensor.init


end Einlean2
