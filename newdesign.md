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
