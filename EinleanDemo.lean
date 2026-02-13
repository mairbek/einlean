import Einlean.Viz

namespace EinleanDemo
open Einlean

structure Rgb where
  r : Nat
  g : Nat
  b : Nat

def c : Dim := dim! 3
def digitW (size : Nat) : Dim := dim! (3 * size)
def digitH (size : Nat) : Dim := dim! (5 * size)

private def pixelOnZero (size x y : Nat) : Bool :=
  let s := size
  let top := y < s ∧ s ≤ x ∧ x < 2 * s
  let mid := 2 * s ≤ y ∧ y < 3 * s ∧ s ≤ x ∧ x < 2 * s
  let bot := 4 * s ≤ y ∧ y < 5 * s ∧ s ≤ x ∧ x < 2 * s
  let lt := s ≤ y ∧ y < 2 * s ∧ x < s
  let lb := 3 * s ≤ y ∧ y < 4 * s ∧ x < s
  let rt := s ≤ y ∧ y < 2 * s ∧ 2 * s ≤ x ∧ x < 3 * s
  let rb := 3 * s ≤ y ∧ y < 4 * s ∧ 2 * s ≤ x ∧ x < 3 * s
  let _ := mid
  top ∨ bot ∨ lt ∨ lb ∨ rt ∨ rb

private def channel (rgb : Rgb) (ch : Nat) : Int :=
  if ch = 0 then Int.ofNat rgb.r
  else if ch = 1 then Int.ofNat rgb.g
  else Int.ofNat rgb.b

def demoSize : Nat := 8
def b : Dim := dim! 6
def b1 : Dim := b.factor! 3
def b2 : Dim := b.factor! 2
def w : Dim := digitW demoSize
def h : Dim := digitH demoSize

private def bgFor (d : Nat) : Rgb :=
  match d % 6 with
  | 0 => { r := 20,  g := 30,  b := 140 }
  | 1 => { r := 150, g := 30,  b := 25 }
  | 2 => { r := 20,  g := 120, b := 35 }
  | 3 => { r := 135, g := 95,  b := 20 }
  | 4 => { r := 85,  g := 25,  b := 125 }
  | _ => { r := 25,  g := 95,  b := 110 }

def ims : Tensor [b, h, w, c] Int :=
  Tensor.ofFn (dims := [b, h, w, c]) fun idx =>
    let bi := idx[0]!
    let y := idx[1]!
    let x := idx[2]!
    let ch := idx[3]!
    let fg : Rgb := { r := 255, g := 255, b := 255 }
    let bg := bgFor (bi % 6)
    if pixelOnZero demoSize x y then
      channel fg ch
    else
      channel bg ch

def img0 : Tensor [h, w, c] Int := ims[0]
def img0T : Tensor [w, h, c] Int := img0.rearrange
def composeBH : Tensor [h, b * w, c] Int := ims.rearrange
def composeBW : Tensor [b * h, w, c] Int := ims.rearrange
def ims2 : Tensor [b1 * b2, h, w, c] Int := ims
def composed : Tensor [b1 * h, b2 * w, c] Int := ims.rearrange
def composed2 : Tensor [b2 * h, b1 * w, c] Int := ims.rearrange

#imgtensor ims
#imgtensor img0
#imgtensor img0T
#imgtensor composeBH
#imgtensor composeBW
#imgtensor composed
#imgtensor composed2

-- ============================================
-- REDUCE DEMOS
-- ============================================

-- Average over batch dimension: [b=6, h, w, c] -> [h, w, c]
-- Blends all 6 digit images into one averaged image
set_option linter.unusedVariables false in
def batchMean : Tensor [h, w, c] Int :=
  ims.reduceBy .mean fun (b, h, w, c) => (h, w, c)

#imgtensor batchMean

-- Max over batch: [b=6, h, w, c] -> [h, w, c]
-- Brightest pixel from any batch image
set_option linter.unusedVariables false in
def batchMax : Tensor [h, w, c] Int :=
  ims.reduceBy .max fun (b, h, w, c) => (h, w, c)

#imgtensor batchMax

-- Reduce + rearrange: average over width -> [b, h, c] then view as image
-- Creates a "column average" strip per batch element
set_option linter.unusedVariables false in
def widthMean : Tensor [b, h, c] Int :=
  ims.reduceBy .mean fun (b, h, w, c) => (b, h, c)

#eval widthMean.shape

end EinleanDemo
