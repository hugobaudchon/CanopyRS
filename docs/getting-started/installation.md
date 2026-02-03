# Installation

## Requirements

- **OS:** Linux (Ubuntu 22.04 recommended)
- **Python:** 3.10+
- **CUDA:** 12.1

## Step-by-step

**1. Clone the repository**

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
git submodule update --init --recursive
```

**2. Create a conda environment**

```bash
conda create -n canopyrs python=3.10 -y
conda activate canopyrs
```

**3. Install GDAL via conda**

```bash
conda install -c conda-forge gdal -y
```

**4. Install CanopyRS**

```bash
pip install -e .
```

⚠️ You will likely encounter this error: `sam2 0.4.1 requires iopath>=0.1.10, but you have iopath 0.1.9 which is incompatible`, which is a conflict between Detectron2 and SAM2 libraries, but it can be ignored and shouldn't impact installation or usage of the pipeline.

## Verify the installation

```bash
python -c "import canopyrs; print('CanopyRS installed successfully')"
```
