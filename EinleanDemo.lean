import Einlean.Viz

namespace EinleanDemo
open Einlean

def testB := dim! 6
def testW := dim! 96
def testH := dim! 96
def testC := dim! 3

private def loadRgbTensor4 (path : System.FilePath)
    (b w h c : Dim) : IO (Tensor [b, w, h, c] Int) := do
  -- TODO: replace this specialized RGB byte loader with a generic tensor I/O API
  -- (e.g. NPY/NPZ support, dtype conversion, and shape inference/validation).
  let bytes ← IO.FS.readBinFile path
  let expected := b.size * w.size * h.size * c.size
  if bytes.size != expected then
    throw <| IO.userError s!"Expected {expected} bytes, got {bytes.size} in {path}"
  let data : Array Int := Id.run do
    let mut out := Array.mkEmpty bytes.size
    for i in [:bytes.size] do
      out := out.push (Int.ofNat (bytes[i]!.toNat))
    return out
  pure <| Tensor.ofArray (dims := [b, w, h, c]) data

def loadIms : IO (Tensor [testB, testW, testH, testC] Int) := do
  loadRgbTensor4 "test_images.rgb" testB testW testH testC

/-- First image in the batch. -/
def loadImg0 : IO (Tensor [testW, testH, testC] Int) := do
  let ims ← loadIms
  pure ims[0]

/-- Transpose width/height: [w, h, c] -> [h, w, c]. -/
def loadImg0T : IO (Tensor [testH, testW, testC] Int) := do
  let img0 ← loadImg0
  pure img0.rearrange

/-- Compose axes [b, h] into one axis: [b, w, h, c] -> [w, (b*h), c]. -/
def loadComposeBH : IO (Tensor [testW, testB * testH, testC] Int) := do
  let ims ← loadIms
  pure ims.rearrange

/-- Compose axes [b, w] into one axis: [b, w, h, c] -> [(b*w), h, c]. -/
def loadComposeBW : IO (Tensor [testB * testW, testH, testC] Int) := do
  let ims ← loadIms
  pure ims.rearrange

#imgtensor_io loadIms
#imgtensor_io loadImg0
#imgtensor_io loadImg0T
#imgtensor_io loadComposeBH
#imgtensor_io loadComposeBW

end EinleanDemo
