import Einlean
import ProofWidgets.Component.HtmlDisplay
import Lean.Data.Json

namespace Einlean

open ProofWidgets

structure VizConfig where
  pixelSize : Nat := 6
  columns : Nat := 3
  gap : Nat := 8
  showBorders : Bool := true
  deriving Repr

private def clampByte (x : Int) : Nat :=
  if x < 0 then
    0
  else if x > 255 then
    255
  else
    Int.toNat x

private def rgbString (r g b : Nat) : String :=
  s!"rgb({r},{g},{b})"

private def jStr (s : String) : Lean.Json :=
  Lean.Json.str s

private def mkRect (x y w h : Nat) (fill : String) : Html :=
  Html.element "rect"
    #[("x", jStr s!"{x}"), ("y", jStr s!"{y}"), ("width", jStr s!"{w}"),
      ("height", jStr s!"{h}"), ("fill", jStr fill)]
    #[]

private def mkOutline (x y w h : Nat) : Html :=
  Html.element "rect"
    #[("x", jStr s!"{x}"), ("y", jStr s!"{y}"), ("width", jStr s!"{w}"),
      ("height", jStr s!"{h}"), ("fill", jStr "none"), ("stroke", jStr "#1f2937"),
      ("strokeWidth", jStr "1")]
    #[]

private def mkSvg (w h : Nat) (children : Array Html) : Html :=
  Html.element "svg"
    #[("xmlns", jStr "http://www.w3.org/2000/svg"),
      ("width", jStr s!"{w}"),
      ("height", jStr s!"{h}"),
      ("viewBox", jStr s!"0 0 {w} {h}")]
    children

private def readRgb {dims : DimList} (t : Tensor dims Int) (indices : Array Nat) (channels : Nat) : Nat × Nat × Nat :=
  let r := clampByte (t.get! (indices.push 0))
  let g := if channels > 1 then clampByte (t.get! (indices.push 1)) else r
  let b := if channels > 2 then clampByte (t.get! (indices.push 2)) else g
  (r, g, b)

def Tensor.toHtmlImage {h w c : Dim} (t : Tensor [h, w, c] Int) (cfg : VizConfig := {}) : Html := Id.run do
  let hN := t.shape[0]!
  let wN := t.shape[1]!
  let cN := t.shape[2]!
  let px := cfg.pixelSize
  let mut children : Array Html := #[]
  for y in [:hN] do
    for x in [:wN] do
      let (r, g, b) := readRgb t #[y, x] cN
      children := children.push (mkRect (x * px) (y * px) px px (rgbString r g b))
  if cfg.showBorders then
    children := children.push (mkOutline 0 0 (wN * px) (hN * px))
  return mkSvg (wN * px) (hN * px) children

def Tensor.toHtmlBatch {b h w c : Dim} (t : Tensor [b, h, w, c] Int) (cfg : VizConfig := {}) : Html := Id.run do
  let bN := t.shape[0]!
  let hN := t.shape[1]!
  let wN := t.shape[2]!
  let cN := t.shape[3]!
  let px := cfg.pixelSize
  let cols := if cfg.columns == 0 then 1 else cfg.columns
  let rows := (bN + cols - 1) / cols
  let tileW := wN * px
  let tileH := hN * px
  let canvasW := cols * tileW + (cols - 1) * cfg.gap
  let canvasH := rows * tileH + (rows - 1) * cfg.gap
  let mut children : Array Html := #[]
  for bi in [:bN] do
    let row := bi / cols
    let col := bi % cols
    let ox := col * (tileW + cfg.gap)
    let oy := row * (tileH + cfg.gap)
    for y in [:hN] do
      for x in [:wN] do
        let (r, g, b) := readRgb t #[bi, y, x] cN
        children := children.push (mkRect (ox + x * px) (oy + y * px) px px (rgbString r g b))
    if cfg.showBorders then
      children := children.push (mkOutline ox oy tileW tileH)
  return mkSvg canvasW canvasH children

def panel (title subtitle : String) (body : Html) : Html :=
  Html.element "div"
    #[]
    #[
      Html.element "h3" #[]
        #[Html.text title],
      Html.element "p" #[]
        #[Html.text subtitle],
      body
    ]

end Einlean
