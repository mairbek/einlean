# Typesafe Einops in Lean — Design Sketch

This document describes a Lean-based design for **typesafe einops-style tensor operations**, where:

- Shapes live in the type system
- Invalid tensor operations fail at **compile time**
- Lambda-based and lambda-free APIs coexist
- Dimension identities (not just sizes) are enforced

The goal is to support:

- `rearrange`, `reduce`, `einsum`
- composed dimensions like `(b, w)` and `(c, h, w)`
- compile-time validation of all shape laws
- einops-like ergonomics

---

## 1. Dimensions (`Dim`)

A `Dim` represents a runtime-sized axis with a **stable identity**.

```lean
structure Dim : Type where
  id   : Nat
  size : Nat
```

Two dims with the same `size` but different `id`s are *not interchangeable*.

---

## 2. Dimension Composition (Atomic but Decomposable)

We want `(b, w)` to behave as:

- a **single atomic dim** in `Tensor [h, (b,w), c]`
- but **decomposable** for indexing and rearranging

```lean
def Dim.mul (a b : Dim) : Dim :=
  { id := a.id * 1_000_003 + b.id
  , size := a.size * b.size }

notation "(" a ", " b ")" => Dim.mul a b
```

Higher-arity composition is represented by nesting:

- `(c, h, w)` ≡ `((c, h), w)`

---

## 3. Canonical Decomposition (`DecompDim`)

Each `Dim` has a canonical *packed representation* used for indexing.

```lean
class DecompDim (d : Dim) where
  Pack : Type
  toLin   : Pack → Fin d.size
  fromLin : Fin d.size → Pack
```

### Primitive Dim

```lean
instance (d : Dim) : DecompDim d where
  Pack := Fin d.size
  toLin := id
  fromLin := id
```

### Composed Dim

```lean
instance (a b : Dim) [DecompDim a] [DecompDim b] :
  DecompDim (a, b) where
  Pack := DecompDim.Pack a × DecompDim.Pack b
  -- toLin / fromLin implement row-major packing
```

This abstraction allows `(b,w)` to behave as both a single axis and a pair of axes.

---

## 4. Shapes and Indices

```lean
abbrev Shape := List Dim
```

### Dependent Index Type

```lean
inductive Idx : Shape → Type
| nil  : Idx []
| cons : DecompDim.Pack d → Idx ds → Idx (d :: ds)
```

This is the type lambdas pattern-match on.

---

## 5. Tensor Core

```lean
def shapeSize : Shape → Nat
| []      => 1
| d :: ds => d.size * shapeSize ds

structure Tensor (ds : Shape) (α : Type) where
  data : Array α
```

Runtime storage is flat; shapes are tracked at the type level.

---

## 6. Rearrangement

### General (Lambda-Based)

```lean
structure Iso (A B : Type) where
  fwd : A → B
  bwd : B → A
  left  : ∀ x, bwd (fwd x) = x
  right : ∀ y, fwd (bwd y) = y
```

```lean
def Tensor.rearrangeBy
  (t : Tensor ds α)
  (p : Iso (Idx ds) (Idx es)) :
  Tensor es α := ...
```

---

## 7. Lambda-Free Rearrange (Type-Driven)

```lean
class CanRearrange (ds es : Shape) where
  iso : Iso (Idx ds) (Idx es)

def Tensor.rearrange [CanRearrange ds es]
  (t : Tensor ds α) : Tensor es α :=
  t.rearrangeBy (CanRearrange.iso)
```

---

## 8. Reduction

```lean
structure Reducer (α : Type) where
  init : α
  step : α → α → α
```

```lean
def Tensor.reduceBy
  (r : Reducer α)
  (t : Tensor ds α)
  (f : Idx ds → Idx es) :
  Tensor es α := ...
```

---

## 9. Einsum

```lean
def Tensor.einsumBy
  (a : Tensor ds₁ α)
  (b : Tensor ds₂ α)
  (out : Idx ds₁ → Idx ds₂ → Idx dsOut) :
  Tensor dsOut α := ...
```

```lean
class CanEinsum2 (ds₁ ds₂ dsOut : Shape) : Prop := ...

def Tensor.einsum
  [CanEinsum2 ds₁ ds₂ dsOut]
  (a : Tensor ds₁ α)
  (b : Tensor ds₂ α) :
  Tensor dsOut α := ...
```

---

## 10. Reshape

```lean
def Tensor.reshape
  (t : Tensor ds α)
  (h : shapeSize ds = shapeSize es) :
  Tensor es α := ...
```

---

## 11. Factorization (`factor!`)

```lean
factor! b1, b2 := b, 2
```

This macro creates new dims, proves size relationships, and enables split/merge rearranges.

---

## 12. Mental Model

**Indices are proofs.  
Shapes are types.  
Einops laws are enforced by elaboration.**
