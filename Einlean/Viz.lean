import Einlean
import ProofWidgets.Component.HtmlDisplay
import Lean.Data.Json
import Lean.Elab.Command

namespace Einlean

open ProofWidgets

structure VizConfig where
  pixelSize : Nat := 1
  columns : Nat := 0
  gap : Nat := 2
  showBorders : Bool := true
  deriving Repr

class ToByte (α : Type) where
  toByte : α → Nat

instance : ToByte Int where
  toByte x :=
    if x < 0 then 0 else if x > 255 then 255 else Int.toNat x

instance : ToByte Nat where
  toByte x := if x > 255 then 255 else x

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
      ("height", jStr s!"{h}"), ("fill", jStr "none"), ("stroke", jStr "#000000"),
      ("strokeWidth", jStr "1")]
    #[]

private def mkSvg (w h : Nat) (children : Array Html) : Html :=
  Html.element "svg"
    #[("xmlns", jStr "http://www.w3.org/2000/svg"),
      ("width", jStr s!"{w}"),
      ("height", jStr s!"{h}"),
      ("viewBox", jStr s!"0 0 {w} {h}")]
    children

private def readRgb {dims : DimList} {α : Type}
    [ToByte α] [Inhabited α]
    (t : Tensor dims α) (indices : Array Nat) : Nat × Nat × Nat :=
  let r := ToByte.toByte (t.get! (indices.push 0))
  let g := ToByte.toByte (t.get! (indices.push 1))
  let b := ToByte.toByte (t.get! (indices.push 2))
  (r, g, b)

private def unsupported (msg : String) : Html :=
  Html.element "p" #[] #[Html.text msg]

def Tensor.toHtmlImage {w h c : Dim} {α : Type}
    [ToByte α] [Inhabited α]
    (t : Tensor [w, h, c] α) (cfg : VizConfig := {}) : Html := Id.run do
  let wN := t.shape[0]!
  let hN := t.shape[1]!
  let cN := t.shape[2]!
  if cN != 3 then
    return unsupported s!"#imgtensor expects c=3 for images, got c={cN}"
  let px := cfg.pixelSize
  let mut children : Array Html := #[]
  for y in [:hN] do
    for x in [:wN] do
      let (r, g, b) := readRgb t #[x, y]
      children := children.push (mkRect (x * px) (y * px) px px (rgbString r g b))
  if cfg.showBorders then
    children := children.push (mkOutline 0 0 (wN * px) (hN * px))
  return mkSvg (wN * px) (hN * px) children

def Tensor.toHtmlBatch {b w h c : Dim} {α : Type}
    [ToByte α] [Inhabited α]
    (t : Tensor [b, w, h, c] α) (cfg : VizConfig := {}) : Html := Id.run do
  let bN := t.shape[0]!
  let wN := t.shape[1]!
  let hN := t.shape[2]!
  let cN := t.shape[3]!
  if cN != 3 then
    return unsupported s!"#imgtensor expects c=3 for batch images, got c={cN}"
  let px := cfg.pixelSize
  let cols := if cfg.columns == 0 then bN else cfg.columns
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
        let (r, g, b) := readRgb t #[bi, x, y]
        children := children.push (mkRect (ox + x * px) (oy + y * px) px px (rgbString r g b))
    if cfg.showBorders then
      children := children.push (mkOutline ox oy tileW tileH)
  return mkSvg canvasW canvasH children

class ImgTensorRenderable (β : Type) where
  toHtml : β → Html

instance {w h c : Dim} {α : Type} [ToByte α] [Inhabited α] :
    ImgTensorRenderable (Tensor [w, h, c] α) where
  toHtml t := t.toHtmlImage

instance {b w h c : Dim} {α : Type} [ToByte α] [Inhabited α] :
    ImgTensorRenderable (Tensor [b, w, h, c] α) where
  toHtml t := t.toHtmlBatch

def imgTensor {β : Type} [ImgTensorRenderable β] (x : β) : Html :=
  ImgTensorRenderable.toHtml x

syntax (name := imgTensorCmd) "#imgtensor " term : command

open Lean Server Elab Command

@[command_elab imgTensorCmd]
def elabImgTensorCmd : CommandElab := fun
  | stx@`(#imgtensor $t:term) => do
    let htX ← liftTermElabM <|
      ProofWidgets.HtmlCommand.evalCommandMHtml <|
      ← ``(ProofWidgets.HtmlEval.eval (Einlean.imgTensor $t))
    let ht ← htX
    liftCoreM <| Widget.savePanelWidgetInfo
      (hash HtmlDisplayPanel.javascript)
      (return json% { html: $(← rpcEncode ht) })
      stx
  | stx => throwError "Unexpected syntax {stx}."

syntax (name := imgTensorIOCmd) "#imgtensor_io " term : command

@[command_elab imgTensorIOCmd]
def elabImgTensorIOCmd : CommandElab := fun
  | stx@`(#imgtensor_io $t:term) => do
    let htX ← liftTermElabM <|
      ProofWidgets.HtmlCommand.evalCommandMHtml <|
      ← ``(ProofWidgets.HtmlEval.eval
            ((($t) >>= fun x => pure (Einlean.imgTensor x)) : IO ProofWidgets.Html))
    let ht ← htX
    liftCoreM <| Widget.savePanelWidgetInfo
      (hash HtmlDisplayPanel.javascript)
      (return json% { html: $(← rpcEncode ht) })
      stx
  | stx => throwError "Unexpected syntax {stx}."

end Einlean
