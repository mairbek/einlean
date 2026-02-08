import Einlean.Viz

namespace EinleanDemo
open Einlean

def b := dim! 6
def w := dim! 32
def h := dim! 24
def c := dim! 3

/-- Synthetic image batch with shape [b, w, h, c]. -/
def ims : Tensor [b, w, h, c] Int :=
  Tensor.ofFn (dims := [b, w, h, c]) fun idx =>
    let bi := idx[0]!
    let x := idx[1]!
    let y := idx[2]!
    let ch := idx[3]!
    match ch with
    | 0 => Int.ofNat ((bi * 41 + x * 6 + y * 3) % 256)
    | 1 => Int.ofNat ((bi * 17 + x * 2 + y * 11) % 256)
    | _ => Int.ofNat ((bi * 29 + x * 9 + y * 5) % 256)

/-- First image in the batch. -/
def img0 : Tensor [w, h, c] Int :=
  Tensor.ofFn (dims := [w, h, c]) fun idx =>
    ims.get! #[0, idx[0]!, idx[1]!, idx[2]!]

/-- Transpose width/height: [w, h, c] -> [h, w, c]. -/
def img0T : Tensor [h, w, c] Int :=
  img0.rearrange

/-- Compose axes [b, h] into one axis: [b, w, h, c] -> [w, (b*h), c]. -/
def composeBH : Tensor [w, b * h, c] Int :=
  ims.rearrange

/-- Compose axes [b, w] into one axis: [b, w, h, c] -> [(b*w), h, c]. -/
def composeBW : Tensor [b * w, h, c] Int :=
  ims.rearrange

#imgtensor ims
#imgtensor img0
#imgtensor img0T
#imgtensor composeBH
#imgtensor composeBW

end EinleanDemo
