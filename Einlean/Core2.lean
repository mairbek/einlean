import Std.Tactic
open Lean

namespace Einlean3

-- --------------------------------------------
-- Runtime dimension
-- --------------------------------------------

structure Dim where
  id   : Nat
  size : Nat
  hpos : 0 < size := by omega
  deriving Repr

def dim (id : Nat) (size : Nat) (hpos : 0 < size := by omega) : Dim :=
  { id := id, size := size, hpos := hpos }

open Lean in
scoped macro "dim!" sz:term : term => do
  let some pos := (← getRef).getPos? | Macro.throwError "dim!: no source position"
  `(dim $(quote pos.byteIdx) $sz)

-- --------------------------------------------
-- Type-level dimension expression
-- --------------------------------------------

inductive DimE where
  | atom (d : Dim)
  | mul (a b : DimE)
  deriving Repr

infixl:70 " * " => DimE.mul

def DimE.eval : DimE → Dim
  | .atom d => d
  | .mul a b =>
    let da := a.eval
    let db := b.eval
    { id := da.id * 1000003 + db.id
    , size := da.size * db.size
    , hpos := Nat.mul_pos da.hpos db.hpos }

def PackOf : DimE → Type
  | .atom d  => Fin d.size
  | .mul a b => PackOf a × PackOf b

abbrev Shape := List DimE

abbrev PIdx : Shape → Type
  | []      => Unit
  | d :: ds => PackOf d × PIdx ds

structure TensorData (α : Type := Int) where
  shape : Array Nat

abbrev Tensor (_ds : Shape) (α : Type := Int) := TensorData α

def shapeOf (ds : Shape) : List Nat :=
  ds.map (fun d => d.eval.size)

def Tensor.init {ds : Shape} {α : Type} : Tensor ds α :=
  { shape := (shapeOf ds).toArray }

def Tensor.rearrangeBy {ds es : Shape} {α : Type}
    (_t : Tensor ds α)
    (_fwd : PIdx ds → PIdx es) : Tensor es α :=
  Tensor.init

-- --------------------------------------------
-- d! macro (DimE atom)
-- --------------------------------------------

open Lean

-- `d! 3` as a DimE atom
scoped macro "d!" sz:term : term =>
  `(DimE.atom (dim! $sz))

syntax (name := decompose2Cmd)
  "decompose2!" ident "as" ident ident ":=" "(" term "," term ")" : command

syntax (name := decompose2LocalCmd)
  "decompose2_local!" ident "as" ident ident ":=" "(" term "," term ")" : command

macro_rules
  | `(decompose2! $b:ident as $b1:ident $b2:ident := ($s1:term, $s2:term)) => do
      `(command|
        macro_rules
          | `($b1) => `(d! $s1)
          | `($b2) => `(d! $s2)
          | `($b)  => `(DimE.mul $b1 $b2)
      )

macro_rules
  | `(decompose2_local! $b:ident as $b1:ident $b2:ident := ($s1:term, $s2:term)) => do
      `(command|
        local macro_rules
          | `($b1) => `(d! $s1)
          | `($b2) => `(d! $s2)
          | `($b)  => `(DimE.mul $b1 $b2)
      )


-- --------------------------------------------
-- DEMOS
-- --------------------------------------------

namespace Test

abbrev b1 : DimE := d! 3
abbrev b2 : DimE := d! 2
abbrev h  : DimE := d! 3
abbrev w  : DimE := d! 4

abbrev b : DimE := b1 * b2

def img2e : Tensor [b, h, w] := Tensor.init

def split : Tensor [b1, b2, h, w] :=
  img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
    (b1i, (b2i, (hi, (wi, ()))))

def split2 : Tensor [h, b] :=
  img2e.rearrangeBy fun (bi, (hi, (_wi, ()))) =>
    (hi, (bi, ()))

def fancy : Tensor [h, b1, w, b2] :=
  img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
    (hi, (b1i, (wi, (b2i, ()))))

end Test

namespace TestDecompose

abbrev tb : DimE := d! 6
abbrev h : DimE := d! 3
abbrev w : DimE := d! 4

-- rewrites terms: tb1 ~> d!2, tb2 ~> d!3, tb ~> tb1*tb2
decompose2! tb as tb1 tb2 := (2, 3)

def img2e : Tensor [tb, h, w] := Tensor.init

def split : Tensor [tb1, tb2, h, w] :=
  img2e.rearrangeBy fun ((tb1i, tb2i), (hi, (wi, ()))) =>
    (tb1i, (tb2i, (hi, (wi, ()))))

def split2 : Tensor [h, tb] :=
  img2e.rearrangeBy fun (bi, (hi, (_wi, ()))) =>
    (hi, (bi, ()))

def fancy : Tensor [h, tb1, w, tb2] :=
  img2e.rearrangeBy fun ((tb1i, tb2i), (hi, (wi, ()))) =>
    (hi, (tb1i, (wi, (tb2i, ()))))

end TestDecompose

namespace TestDecomposeScoped

abbrev b : DimE := d! 6
abbrev h : DimE := d! 3
abbrev w : DimE := d! 4

section

decompose2_local! b as b1 b2 := (2, 3)

def img2e : Tensor [b, h, w] := Tensor.init

def split : Tensor [b1, b2, h, w] :=
  img2e.rearrangeBy fun ((b1i, b2i), (hi, (wi, ()))) =>
    (b1i, (b2i, (hi, (wi, ()))))

def split2 : Tensor [h, b] :=
  img2e.rearrangeBy fun (bi, (hi, (_wi, ()))) =>
    (hi, (bi, ()))

end

def outsideSection : Tensor [b, h, w] := Tensor.init


end TestDecomposeScoped

end Einlean3
