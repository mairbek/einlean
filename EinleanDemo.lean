import Einlean.Viz
import Einlean.TestImagesData

namespace EinleanDemo
open Einlean

def ims : Tensor [testB, testW, testH, testC] Int :=
  testImages

/-- First image in the batch. -/
def img0 : Tensor [testW, testH, testC] Int :=
  Tensor.ofFn (dims := [testW, testH, testC]) fun idx =>
    ims.get! #[0, idx[0]!, idx[1]!, idx[2]!]

/-- Transpose width/height: [w, h, c] -> [h, w, c]. -/
def img0T : Tensor [testH, testW, testC] Int :=
  img0.rearrange

/-- Compose axes [b, h] into one axis: [b, w, h, c] -> [w, (b*h), c]. -/
def composeBH : Tensor [testW, testB * testH, testC] Int :=
  ims.rearrange

/-- Compose axes [b, w] into one axis: [b, w, h, c] -> [(b*w), h, c]. -/
def composeBW : Tensor [testB * testW, testH, testC] Int :=
  ims.rearrange

#imgtensor ims
#imgtensor img0
#imgtensor img0T
#imgtensor composeBH
#imgtensor composeBW

end EinleanDemo
