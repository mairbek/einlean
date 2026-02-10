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
def b1 : Dim := dim! 3
def b2 : Dim := dim! 2
def b : Dim := b1 * b2
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

def ims : Tensor [b, w, h, c] Int :=
  Tensor.ofFn (dims := [b, w, h, c]) fun idx =>
    let bi := idx[0]!
    let x := idx[1]!
    let y := idx[2]!
    let ch := idx[3]!
    let fg : Rgb := { r := 255, g := 255, b := 255 }
    let bg := bgFor (bi % 6)
    if pixelOnZero demoSize x y then
      channel fg ch
    else
      channel bg ch

def img0 : Tensor [w, h, c] Int := ims[0]
def img0T : Tensor [h, w, c] Int := img0.rearrange
def composeBH : Tensor [w, b * h, c] Int := ims.rearrange
def composeBW : Tensor [b * w, h, c] Int := ims.rearrange

/-- Equivalent to einops: (b1 b2) w h c -> b1 b2 w h c. -/
def splitB : Tensor [b1, b2, w, h, c] Int := ims.reshape

/-- Equivalent to einops: (b1 b2) w h c -> (b1 w) (b2 h) c. -/
def compose3x2Split : Tensor [b1 * w, b2 * h, c] Int := splitB.rearrange
def compose2x3Split : Tensor [b2 * w, b1 * h, c] Int := splitB.rearrange

#imgtensor ims
#imgtensor img0
#imgtensor img0T
#imgtensor composeBH
#imgtensor composeBW
#imgtensor compose3x2Split
#imgtensor compose2x3Split

end EinleanDemo
