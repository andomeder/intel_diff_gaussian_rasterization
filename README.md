# Intel XPU Differential Gaussian Rasterization

A high-performance implementation of Differential Gaussian Rasterization for Intel Arc GPUs using SYCL/DPC++. This project provides a drop-in replacement for CUDA-based implementations, optimized for Intel XPU architecture.

## Overview

This implementation is based on the paper **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** and provides:

- **SYCL/DPC++ kernels** optimized for Intel Arc GPUs (tested on B580)
- **PyTorch integration** for seamless use in training and inference pipelines
- **Forward pass implementation** with tile-based rendering optimizations
- **Production-ready** performance for real-time rendering

## Hardware Requirements

- **GPU**: Intel Arc GPU (A-series, B-series) with Level Zero support
- **Drivers**: Native Support on Linux Kernel 6.7+. Ensure current Mesa.
- **Memory**: 4GB+ VRAM recommended

### Tested Configuration
- **GPU**: Intel Arc B580 (12GB VRAM, 20 Xe-cores)
- **OS**: Ubuntu 25.04 LTS / Arch Linux
- **Compiler**: Intel DPC++ 2025.0.4 (Arch) / PyTorch XPU 2.8.0 (Ubuntu)

## Performance Benchmarks

Tested on **Intel Arc B580** at 800x800 resolution:

| Gaussians | Frame Time | FPS    | Notes |
|-----------|------------|--------|-------|
| 100       | 6.11ms     | 163.6  | Minimal scene |
| 1,000     | 5.84ms     | 171.3  | Typical object |
| 10,000    | 5.85ms     | 170.9  | Complex object |
| 50,000    | 7.02ms     | 142.4  | Dense scene |
| 100,000   | 11.08ms    | 90.2   | Very dense |
| 150,000   | 11.57ms    | 86.4   | Stress test |

**Key Findings:**
- Consistent ~170 FPS for scenes up to 10K Gaussians
- Graceful degradation for complex scenes
- Memory bandwidth saturates around 150K Gaussians
- Forward-only rendering (backward pass not yet implemented)

### Comparison with CPU Implementation

| Implementation | 1K Gaussians | 10K Gaussians | 100K Gaussians | 150K Gaussians |
|----------------|--------------|---------------|----------------|----------------|
| C CPU (Pure)   | 4.93 ms      | 19.12 ms      | 173.30 ms      | 256.83 ms      |
| Intel Arc B580 | 5.84 ms      | 5.85 ms       | 11.08 ms       | 11.57 ms       |
| **Speedup**    | **0.84×** (≈ same) | **3.27×** | **15.6×** | **22.2×** |

## Installation

### Method 1: Pre-built Binary Wheel (Recommended - Ubuntu/Arch)

Download the pre-built wheel from the [Releases](https://github.com/andomeder/intel_diff_gaussian_rasterization/releases) page:

```bash
# 1. Install PyTorch XPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# OR for CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the pre-built wheel
pip install intel_diff_gaussian_rasterization-0.1.0-cp310-cp310-linux_x86_64.whl

# 3. Set up environment (Ubuntu only)
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

# 4. Test installation
python -c "import intel_diff_gaussian_rasterization; print('✓ Success')"
```

**Platform Compatibility:**
- **Python**: 3.10
- **OS**: Ubuntu 22.04/24.04/25.04, Arch Linux
- **Architecture**: x86_64 only
- **GPU**: Intel Arc (A/B-series) or compatible integrated graphics

### Method 2: From Source Tarball (No Compilation Required)

Download the source tarball from [Releases](https://github.com/andomeder/intel_diff_gaussian_rasterization/releases):

```bash
# 1. Install PyTorch XPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# 2. Extract tarball
tar xzf intel_diff_gaussian_rasterization-py3.10-20251107.tar.gz
cd intel_diff_gaussian_rasterization

# 3. Set up environment
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 4. Test
python -c "import intel_diff_gaussian_rasterization; print('✓ Success')"
```

Or use the provided helper script:
```bash
source setup_env.sh
python your_script.py
```

### Method 3: Build from Source (Arch Linux)

**Prerequisites:**
- Arch Linux
- Intel oneAPI DPC++ compiler
- PyTorch (CPU or XPU version)

```bash
# 1. Install dependencies
sudo pacman -S intel-oneapi-dpcpp-cpp intel-oneapi-compiler-shared-runtime
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Clone repository
git clone https://github.com/andomeder/intel_diff_gaussian_rasterization.git
cd intel_diff_gaussian_rasterization

# 3. Build
bash build.sh

# This creates:
# - intel_diff_gaussian_rasterization/_C.cpython-310-*.so (compiled extension)
# - intel_diff_gaussian_rasterization-py3.10-*.tar.gz (distributable tarball)
# - dist/*.whl (binary wheel, if you run build_wheel.sh)
```


## Quick Start After Installation

```python
import torch
import intel_diff_gaussian_rasterization as idgr

# Move tensors to Intel GPU
device = torch.device("xpu")  # Intel Arc GPU

# Your Gaussian splatting code here...
rasterizer = idgr.GaussianRasterizer(settings)
image, radii = rasterizer(
    means3D=means.to(device),
    means2D=means2d.to(device),
    # ... other parameters
)
```

## System Requirements

### Minimum
- **OS**: Ubuntu 22.04+ or Arch Linux
- **Python**: 3.8-3.13
- **PyTorch**: 2.0.0+
- **GPU**: Intel Arc A380 or newer (4GB VRAM)

### Recommended
- **OS**: Ubuntu 24.04 or Arch Linux (latest)
- **Python**: 3.10 (best compatibility)
- **PyTorch**: 2.8.0+xpu (for GPU acceleration)
- **GPU**: Intel Arc B580 (12GB VRAM)
- **RAM**: 16GB+

### Build System Requirements (for building from source)
- **OS**: Arch Linux
- **Compiler**: Intel oneAPI DPC++ 2025.0.4+
- **CMake**: 3.20+
- **Disk**: 5GB free (for Intel oneAPI installation)

## Downloading Pre-built Packages

Visit the [Releases](https://github.com/andomeder/intel_diff_gaussian_rasterization/releases/latest) page to download:

- **Binary Wheels** (`.whl`): For `pip install`, no compilation needed
  - `intel_diff_gaussian_rasterization-0.1.0-cp310-cp310-linux_x86_64.whl` (Python 3.10)

- **Source Tarballs** (`.tar.gz`): Pre-compiled `.so` file included
  - `intel_diff_gaussian_rasterization-py3.10-20251107.tar.gz`

Choose the package matching your Python version.

## Troubleshooting Installation

### Issue: "No module named 'intel_diff_gaussian_rasterization'"

**Solution**: Ensure `PYTHONPATH` is set if using tarball:
```bash
export PYTHONPATH="/path/to/intel_diff_gaussian_rasterization:${PYTHONPATH}"
```

### Issue: "libsycl.so.8: cannot open shared object file"

**Solution**: Set `LD_LIBRARY_PATH` to PyTorch's libraries:
```bash
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
```

### Issue: "LIBUR_LOADER version not found"

**Solution**: Run the above command or Remove conflicting Intel oneAPI installation on Ubuntu(**ONLY IF YOU NEED TO**):
```bash
# remove oneapi to unset vars(you can re-install later)
sudo rm -rf /opt/intel/oneapi
# Then restart your shell
```

### Issue: "pip install" fails with "error: unrecognized command-line option '-fsycl'"

**Explanation**: Building from source requires the Intel DPC++ compiler, which is only properly set up on Arch Linux in our build system. 

**Solution**: Use a pre-built wheel or tarball from Releases instead of building from source.

## Usage

### Basic Example

```python
import torch
import intel_diff_gaussian_rasterization as idgr

# Setup rasterization settings
raster_settings = idgr.GaussianRasterizationSettings(
    image_height=800,
    image_width=800,
    tanfovx=0.5,
    tanfovy=0.5,
    bg=torch.tensor([0.0, 0.0, 0.0]).xpu(),
    scale_modifier=1.0,
    viewmatrix=view_matrix.xpu(),
    projmatrix=proj_matrix.xpu(),
    sh_degree=3,
    campos=camera_position.xpu(),
    prefiltered=False,
    debug=False
)

# Create rasterizer
rasterizer = idgr.GaussianRasterizer(raster_settings)

# Rasterize Gaussians
rendered_image, radii = rasterizer(
    means3D=gaussian_means.xpu(),
    means2D=gaussian_means_2d.xpu(),
    opacities=gaussian_opacities.xpu(),
    shs=spherical_harmonics.xpu(),
    scales=gaussian_scales.xpu(),
    rotations=gaussian_rotations.xpu()
)
```

### Drop-in Replacement for CUDA Version

```python
# Instead of:
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# Use:
from intel_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# Replace .cuda() with .xpu()
tensor = tensor.xpu()  # Instead of tensor.cuda()
```

### Helper Scripts

The package includes wrapper scripts for proper environment setup:

```bash
# Run Python with correct library paths
./run_with_intel_raster.sh your_script.py

# Or source the environment
source setup_env.sh
python your_script.py
```

## Architecture Details

### Build System

This was built using a two-stage build process:

1. **Built on Arch Linux** with Intel oneAPI DPC++ compiler 2025.0.4
2. **Deployed on Ubuntu** using PyTorch XPU's bundled SYCL runtime

This avoided version conflicts between oneAPI and PyTorch on the deployment side.

### Kernel Pipeline

1. **Preprocessing Kernel** (parallel per Gaussian)
   - Frustum culling
   - 3D to 2D projection via perspective transformation
   - 3D covariance → 2D covariance projection
   - Spherical harmonics to RGB conversion

2. **Binning Phase**
   - Tile assignment (16x16 pixel tiles)
   - Depth-based sorting (currently CPU-based)
   - Tile range identification

3. **Rendering Kernel** (parallel per pixel)
   - Tile-based processing with shared memory
   - Alpha blending with early ray termination
   - Sub-group optimizations for Intel Arc

### Optimizations

- **Work-group size**: 16×16 (256 work-items) for optimal Arc GPU occupancy
- **Memory coalescing**: Aligned memory access patterns
- **Shared memory**: Cooperative loading of Gaussian data per tile
- **Early termination**: Skip pixels when accumulated alpha < 0.0001
- **RPATH**: Binaries link directly to PyTorch's SYCL libraries

## Current Limitations

- ✅ Forward pass fully implemented and tested
- ❌ Backward pass not implemented (training not supported)
- ⚠️  Sorting done on CPU (GPU radix sort planned)
- ⚠️  Device may reset with >150K Gaussians (driver stability)
- ⚠️  WebGPU export not yet available

## Roadmap

### Short Term
- [ ] Optimize tile sorting with GPU-based radix sort
- [ ] Improve memory management for large scenes
- [ ] Add batch rendering support

### Medium Term
- [ ] Implement backward pass for training
- [ ] Multi-GPU support via SYCL device selection
- [ ] Integrate with Intel Extension for PyTorch (IPEX)

### Long Term
- [ ] WebGPU SPIR-V export for web deployment
- [ ] Numba-dpex alternative implementation
- [ ] Extended platform support (AMD via hipSYCL)

## More Troubleshooting

### GPU Not Detected

```bash
# Check SYCL devices
python -c "import torch; print(torch.xpu.is_available()); print(torch.xpu.device_count())"

# Check Level Zero
ls /usr/lib/x86_64-linux-gnu/libze_loader.so*

# Verify drivers
clinfo | grep Intel
```

### Library Version Conflicts

If you see `LIBUR_LOADER` version errors:

```bash
# Remove any system-installed oneAPI (on Ubuntu)
sudo rm -rf /opt/intel/oneapi

# Ensure PyTorch's libraries are first in path
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

### Device Lost Errors

The "Device Lost" error with very large scenes (>150K Gaussians) is a known driver limitation. Workarounds:

1. Reduce scene complexity
2. Render in tiles
3. Update to latest Intel GPU drivers

## Citation

```bibtex
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

This project inherits the license from the original Gaussian Splatting implementation. See LICENSE.md for details.

## Acknowledgments

- Original [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by INRIA GRAPHDECO research group
- C CPU implementation by [MrSecant](https://github.com/MrSecant/diff-gaussian-rasterization)
- Intel oneAPI and SYCL teams
- PyTorch XPU backend team

## Contributing

Contributions welcome! Priority areas:
1. Backward pass implementation
2. GPU-based sorting algorithms
3. Performance profiling and optimization
4. Documentation and examples

For questions or issues, please open a GitHub issue.

---

**Project Status**: Production-ready for inference, research-stage for training.  
**Maintainer**: William Obino  
**Last Updated**: November 2025
