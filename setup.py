from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import sys
import subprocess
import torch

print("="*60)
print("Intel XPU Gaussian Rasterization Setup")
print("="*60)

# Get PyTorch's include and library paths
torch_path = os.path.dirname(torch.__file__)
torch_lib = os.path.join(torch_path, 'lib')
torch_include = os.path.join(torch_path, 'include')

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch_path}")
print(f"PyTorch lib: {torch_lib}")

# Check for Intel oneAPI icpx compiler
def check_icpx():
    try:
        result = subprocess.run(['icpx', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Found icpx: {result.stdout.split()[0]}")
            return True
    except FileNotFoundError:
        pass
    print("⚠ icpx not found, will use system compiler")
    return False

has_icpx = check_icpx()

current_dir = os.path.dirname(os.path.abspath(__file__))
sycl_dir = os.path.join(current_dir, 'sycl_rasterizer')

# Compiler flags
extra_compile_args = {
    'cxx': [
        '-O3',
        '-ffast-math',
        '-march=native',
        '-std=c++17',
        '-fPIC',
        '-Wall',
        '-Wno-nan-infinity-disabled',
    ]
}

# Add SYCL flags only if using icpx
if has_icpx:
    extra_compile_args['cxx'].append('-fsycl')
    print("✓ SYCL compilation enabled")
else:
    print("⚠ Building without SYCL (CPU-only mode)")

# Linker flags - use PyTorch's libraries
extra_link_args = [
    f'-L{torch_lib}',
    '-Wl,-rpath,' + torch_lib,  # RPATH to find PyTorch's libs at runtime
]

if has_icpx:
    extra_link_args.extend([
        '-fsycl',
        '-lsycl',
    ])

# Source files
sources = [
    'rasterize_points.cpp',
]

# Only add SYCL sources if we have icpx
if has_icpx:
    sources.extend([
        'sycl_rasterizer/forward.dp.cpp',
        'sycl_rasterizer/rasterizer_impl.dp.cpp',
        'sycl_rasterizer/render.dp.cpp',
    ])
else:
    print("⚠ SYCL sources not included - extension will fail to link")
    print("  Please install Intel oneAPI and source setvars.sh")

# Include directories
include_dirs = [
    os.path.join(current_dir, 'sycl_rasterizer'),
]

# Create extension module
ext_modules = [
    CppExtension(
        name='intel_diff_gaussian_rasterization._C',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    )
]

print("\nBuild configuration:")
print(f"  Sources: {len(sources)} files")
print(f"  Compile flags: {extra_compile_args['cxx'][:5]}...")
print(f"  Link flags: {extra_link_args[:3]}...")
print("="*60)

setup(
    name='intel_diff_gaussian_rasterization',
    version='0.1.0',
    author='William Obino',
    description='Intel XPU Gaussian Rasterization for 3D Gaussian Splatting',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=['intel_diff_gaussian_rasterization'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    install_requires=[
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
    classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
