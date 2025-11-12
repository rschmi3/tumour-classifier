{
  description = "Python development environment for brain tumor classification project";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # CUDA shell with FHS-style environment
        cudaShell = pkgs.mkShell {
          name = "cuda-python-env";
          packages = with pkgs; [
            python311
            python311Packages.pip
            python311Packages.virtualenv
            stdenv.cc.cc.lib
            zlib
            glib
            libGL
            glibc
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.glib
                pkgs.libGL
                pkgs.glibc
              ]
            }:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"

            export PIP_PREFIX="$PWD/.venv-cuda"
            export PYTHONPATH="$PIP_PREFIX/lib/python3.11/site-packages:$PWD:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"

            if [ ! -d .venv-cuda ]; then
              echo "Creating CUDA virtual environment..."
              python -m venv .venv-cuda
              source .venv-cuda/bin/activate
              
              echo "Installing PyTorch with CUDA 12.1 support..."
              pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
              
              echo "Installing packages..."
              pip install -r requirements.txt
              
              echo ""
              echo "✓ CUDA environment ready!"
            else
              source .venv-cuda/bin/activate
            fi

            echo "Testing CUDA availability..."
            python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not yet installed"
          '';
        };

        # ROCm shell with FHS-style environment
        rocmShell = pkgs.mkShell {
          name = "rocm-python-env";
          packages = with pkgs; [
            python311
            python311Packages.pip
            python311Packages.virtualenv
            stdenv.cc.cc.lib
            zlib
            zstd
            glib
            libGL
            glibc
            # ROCm packages
            rocmPackages.clr
            rocmPackages.rocm-core
            rocmPackages.rocm-runtime
            rocmPackages.rocm-device-libs
            rocmPackages.rocm-smi
            rocmPackages.rocminfo
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.zstd
                pkgs.glib
                pkgs.libGL
                pkgs.glibc
                pkgs.rocmPackages.clr
                pkgs.rocmPackages.rocm-core
                pkgs.rocmPackages.rocm-runtime
                pkgs.rocmPackages.rocm-device-libs
              ]
            }:$LD_LIBRARY_PATH"

            export HSA_OVERRIDE_GFX_VERSION=11.0.3
            export ROCM_PATH="${pkgs.rocmPackages.clr}"
            export GPU_DEVICE_ORDINAL=0
            export PIP_PREFIX="$PWD/.venv-rocm"
            export PYTHONPATH="$PIP_PREFIX/lib/python3.11/site-packages:$PWD:$PYTHONPATH"
            export PATH="$PIP_PREFIX/bin:$PATH"

            if [ ! -d .venv-rocm ]; then
              echo "Creating ROCm virtual environment..."
              python -m venv .venv-rocm
              source .venv-rocm/bin/activate
              
              echo "Installing PyTorch with ROCm 6.1 support..."
              pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
              
              echo "Installing packages..."
              pip install -r requirements.txt
              
              echo ""
              echo "✓ ROCm environment ready!"
            else
              source .venv-rocm/bin/activate
            fi

            echo "GPU Information:"
            echo "Device: AMD Radeon 780M (gfx1103)"
            rocm-smi --showproductname 2>/dev/null || true

            echo ""
            echo "Testing ROCm availability..."
            python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not yet installed"
          '';
        };
      in
      {
        devShells = {
          cuda = cudaShell;
          rocm = rocmShell;
          # Default to CUDA
          default = cudaShell;
        };
      }
    );
}
