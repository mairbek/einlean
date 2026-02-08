import Einlean.Viz

namespace Einlean

def vb := dim! 6
def vw := dim! 24
def vh := dim! 24
def vc := dim! 3

def sampleBatch : Tensor [vb, vw, vh, vc] Int :=
  Tensor.ofFn (dims := [vb, vw, vh, vc]) fun idx =>
    let b := idx[0]!
    let x := idx[1]!
    let y := idx[2]!
    let chan := idx[3]!
    match chan with
    | 0 => Int.ofNat ((b * 23 + y * 9 + x * 5) % 256)
    | 1 => Int.ofNat ((b * 41 + y * 3 + x * 11) % 256)
    | _ => Int.ofNat ((b * 17 + y * 13 + x * 7) % 256)

def sampleImage : Tensor [vw, vh, vc] Int :=
  Tensor.ofFn (dims := [vw, vh, vc]) fun idx =>
    let x := idx[0]!
    let y := idx[1]!
    let chan := idx[2]!
    match chan with
    | 0 => Int.ofNat ((y * 8 + x * 3) % 256)
    | 1 => Int.ofNat ((y * 2 + x * 12) % 256)
    | _ => Int.ofNat ((y * 15 + x * 4) % 256)

#imgtensor sampleImage

#imgtensor sampleBatch

end Einlean
