{
  description = "Development shell for cpp-vulnerability-detection";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      runtimeLibs = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.openssl
        pkgs.libffi
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python313
          pkgs.uv
          pkgs.cacert
          pkgs.gcc
          pkgs.git
          pkgs.pkg-config
          pkgs.openssl
          pkgs.zlib
          pkgs.libffi
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${runtimeLibs}:''${LD_LIBRARY_PATH:-}"
          export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
          export UV_LINK_MODE=copy
          export UV_PROJECT_ENVIRONMENT=.venv
          export PYTHONNOUSERSITE=1
        '';
      };
    };
}
