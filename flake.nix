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
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python312.override {
          self = python;

          packageOverrides =
            self: super:
            let
              tf = super.tensorflowWithCuda;

              # Helper function to override tensorflow dependency
              overrideTensorflow =
                pkg:
                (pkg.override {
                  tensorflow = tf;
                }).overridePythonAttrs
                  (old: {
                    doCheck = false;
                  });

              # Helper function to disable checks
              disableCheck =
                pkg:
                pkg.overridePythonAttrs (old: {
                  doCheck = false;
                });
              noCheckPkgs = [
              ];
              gpuTensorflowPkgs = [
                "keras"
                "tf2onnx"
              ];

            in
            pkgs.lib.genAttrs noCheckPkgs (name: disableCheck super.${name})
            // pkgs.lib.genAttrs gpuTensorflowPkgs (name: overrideTensorflow super.${name})
            // {
              xgboost = super.xgboost.override {
                xgboost = pkgs.xgboost.override { cudaSupport = true; };
              };
            };
        };

        classifierPyPkgs = with python.pkgs; [
          kagglehub
          keras
          matplotlib
          numpy
          opencv4
          pandas
          scikit-image
          scikit-learn
          seaborn
          tensorflowWithCuda
          xgboost
        ];

        classifierPkg = python.pkgs.buildPythonPackage {
          pname = "tumour-classifier";
          version = "0.1.0";
          src = ./src;
          pyproject = true;

          build-system = with python.pkgs; [
            setuptools
          ];

          dependencies = classifierPyPkgs;

          # Skip the runtime dependency check since opencv4 provides opencv-python functionality
          pythonRemoveDeps = [ "opencv-python" ];

          # Now this wrapping will work because setuptools creates the script
          postInstall = ''
            wrapProgram $out/bin/tumour-classifier \
              --prefix PYTHONPATH : "$out/${python.sitePackages}"
          '';
          nativeBuildInputs = [ pkgs.makeWrapper ];
        };

        # CUDA shell
        cudaShell = pkgs.mkShell {
          name = "tumour-classifier-cuda-shell";
          packages = with pkgs; [
            classifierPkg
            python.pkgs.ipython
            python.pkgs.pip

            cudaPackages.cuda_nvcc # Provides ptxas compiler
            cudaPackages.cudatoolkit # Full CUDA toolkit
            cudaPackages.cudnn # cuDNN for TensorFlow

            glib
            glibc
            libGL
            stdenv.cc.cc.lib
            zlib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.glib
                pkgs.libGL
                pkgs.glibc
                pkgs.cudaPackages.cuda_nvcc # Provides ptxas compiler
                pkgs.cudaPackages.cudatoolkit # Full CUDA toolkit
                pkgs.cudaPackages.cudnn # cuDNN for TensorFlow
              ]
            }:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"

            echo "Python environment ready!"
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            python --version

            echo "GPU Information:"
            nvidia-smi 2>/dev/null || true

            echo "Testing CUDA availability..."
            python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'Built with CUDA: {tf.test.is_built_with_cuda()}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); [print(f'  {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "TensorFlow not yet installed"'';
        };

        rocmShell = pkgs.mkShell {
          name = "tumour-classifier-rocm-shell";
          packages = with pkgs; [
            classifierPkg

            glib
            glibc
            libGL
            stdenv.cc.cc.lib
            zlib
            zstd

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

            echo "Python environment ready!"
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            python --version

            echo "GPU Information:"
            rocm-smi --showproductname 2>/dev/null || true

            echo ""
            echo "Testing ROCm availability..."
            python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'Built with CUDA: {tf.test.is_built_with_cuda()}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); [print(f'  {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "TensorFlow not yet installed"
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
        hydraJobs = {
          cudaShell = cudaShell;
          rocmShell = rocmShell;
        };

      }
    );
}
