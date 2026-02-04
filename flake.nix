{
  description = "Einlean - Compile-time verifiable tensor operations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.elan
          ];

          shellHook = ''
            echo "Einlean development environment"
            echo "Run 'lake build' to build the project"
          '';
        };
      }
    );
}
