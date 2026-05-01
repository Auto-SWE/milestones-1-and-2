{
  description = "Development shell for cpp-vulnerability-detection";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv
              pkgs.cacert
              pkgs.git
              pkgs.pkg-config
              pkgs.openssl
              pkgs.zlib
              pkgs.libffi
            ];

            shellHook = ''
              unset LD_LIBRARY_PATH

              export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              export UV_LINK_MODE=copy
              export UV_PROJECT_ENVIRONMENT=.venv
              export PYTHONNOUSERSITE=1
            '';
          };
        });
    };
}
