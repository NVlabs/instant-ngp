{
  description = "INGP dev env";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            config.cudaVersion = "12";
          };
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            gcc
            gdb
            cmake
            pkg-config
            binutils
            zlib

            xorg.libX11.dev
            xorg.libXi.dev
            xorg.libXrandr.dev
            xorg.libXinerama.dev
            xorg.libXcursor.dev
            xorg.libXext.dev
            xorg.libXfixes.dev
            xorg.libXrender.dev
            libGL
            glew

            vulkan-loader
            vulkan-headers
            vulkan-validation-layers
            vulkan-extension-layer
            vulkan-tools

            python3
            stdenv.cc.cc.lib

            cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_nvtx
          ];

          shellHook = ''
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export CLANGD_CUDA_INCLUDE="${pkgs.cudatoolkit}"

            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
            export VULKAN_SDK="${pkgs.vulkan-loader}"

            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d:${pkgs.vulkan-extension-layer}/share/vulkan/explicit_layer.d"
            export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json"
          '';
        };
      }
    );
}
