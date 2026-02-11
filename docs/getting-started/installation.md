# Installation

## Requirements

- **OS:** Linux (Ubuntu 22.04 recommended)
- **Python:** 3.10
- **CUDA:** 12.6 — You can install CUDA by following the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). The command `nvcc --version` should show version 12.6.

## Step-by-step

**1. Clone the repository**

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
```

**2. Create a conda environment with mamba**

```bash
conda create -n canopyrs -c conda-forge python=3.10 mamba
conda activate canopyrs
```

**3. Install GDAL via mamba**

```bash
mamba install gdal=3.6.2 -c conda-forge
```

**4. Install PyTorch with CUDA 12.6 support**

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

**5. Initialize submodules**

```bash
git submodule update --init --recursive
```

**6. Install CanopyRS and dependencies**

```bash
python -m pip install -e .
python -m pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
```

## Known issues

You will likely encounter this error during installation:

```
sam2 0.4.1 requires iopath>=0.1.10, but you have iopath 0.1.9 which is incompatible
```

This is a conflict between Detectron2 and SAM2 libraries, but it can be ignored and should not impact installation or usage of the pipeline.

## Verify the installation

```bash
python -c "import canopyrs; print('CanopyRS installed successfully')"
```
