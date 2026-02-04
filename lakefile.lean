import Lake
open Lake DSL

package einlean where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib Einlean where
  roots := #[`Einlean]
