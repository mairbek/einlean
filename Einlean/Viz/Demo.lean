import Einlean.Viz

namespace Einlean

def vb := dim! 6
def vh := dim! 24
def vw := dim! 24
def vc := dim! 3

def sampleBatch : Tensor [vb, vh, vw, vc] Int :=
  Tensor.ofFn (dims := [vb, vh, vw, vc]) fun idx =>
    let b := idx[0]!
    let y := idx[1]!
    let x := idx[2]!
    let chan := idx[3]!
    match chan with
    | 0 => Int.ofNat ((b * 23 + y * 9 + x * 5) % 256)
    | 1 => Int.ofNat ((b * 41 + y * 3 + x * 11) % 256)
    | _ => Int.ofNat ((b * 17 + y * 13 + x * 7) % 256)

def sampleImage : Tensor [vh, vw, vc] Int :=
  Tensor.ofFn (dims := [vh, vw, vc]) fun idx =>
    let y := idx[0]!
    let x := idx[1]!
    let chan := idx[2]!
    match chan with
    | 0 => Int.ofNat ((y * 8 + x * 3) % 256)
    | 1 => Int.ofNat ((y * 2 + x * 12) % 256)
    | _ => Int.ofNat ((y * 15 + x * 4) % 256)

def sampleImageT : Tensor [vw, vh, vc] Int :=
  sampleImage.rearrange

def mergedBH : Tensor [vb * vh, vw, vc] Int :=
  sampleBatch.rearrange

def mergedBW : Tensor [vh, vb * vw, vc] Int :=
  sampleBatch.rearrange

def batchCfg : VizConfig :=
  { pixelSize := 4, columns := 3, gap := 8, showBorders := true }

def imageCfg : VizConfig :=
  { pixelSize := 6, columns := 1, gap := 8, showBorders := true }

#html panel
  "Original"
  "Tensor [b, h, w, c] rendered as a batch grid"
  (sampleBatch.toHtmlBatch batchCfg)

#html panel
  "Transpose"
  "Tensor [h, w, c] -> [w, h, c]"
  (sampleImageT.toHtmlImage imageCfg)

#html panel
  "Compose axes"
  "Tensor [b, h, w, c] -> [(b h), w, c]"
  (mergedBH.toHtmlImage { pixelSize := 2, showBorders := true })

#html panel
  "Compose axes (order matters)"
  "Tensor [b, h, w, c] -> [h, (b w), c]"
  (mergedBW.toHtmlImage { pixelSize := 2, showBorders := true })

end Einlean
