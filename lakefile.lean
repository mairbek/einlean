import Lake
open Lake DSL

package einlean where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

require proofwidgets from git
  "https://github.com/leanprover-community/ProofWidgets4" @ "v0.0.47"

@[default_target]
lean_lib Einlean where
  roots := #[`Einlean, `Einlean.Viz, `Einlean.Viz.Demo, `Einlean.ForwardPassDemo, `EinleanDemo, `EinleanTest]
