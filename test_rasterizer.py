#!/usr/bin/env python3
"""
Test script for Intel XPU Gaussian Rasterization
"""

import torch
import numpy as np
import time
import sys

try:
    import intel_diff_gaussian_rasterization as idgr
    print("✓ Successfully imported intel_diff_gaussian_rasterization")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def create_test_gaussians(num_gaussians=1000, device='cpu'):
    """Create synthetic test Gaussians"""
    torch.manual_seed(42)
    
    # Random positions in a sphere
    means3D = torch.randn(num_gaussians, 3, device=device, dtype=torch.float32) * 2.0
    means2D = torch.randn(num_gaussians, 3, device=device, dtype=torch.float32)  # Placeholder
    
    # Random colors (RGB)
    colors = torch.rand(num_gaussians, 3, device=device, dtype=torch.float32)
    
    # Random opacities
    opacities = torch.rand(num_gaussians, 1, device=device, dtype=torch.float32) * 0.9 + 0.1
    
    # Random scales
    scales = torch.rand(num_gaussians, 3, device=device, dtype=torch.float32) * 0.1 + 0.05
    
    # Random rotations (quaternions)
    rotations = torch.randn(num_gaussians, 4, device=device, dtype=torch.float32)
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)
    
    return means3D, means2D, colors, opacities, scales, rotations


def create_camera_matrices(device='cpu'):
    """Create simple camera matrices"""
    # View matrix (identity)
    viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
    
    # Projection matrix (simple perspective)
    fov = 0.8
    aspect = 1.0
    near = 0.1
    far = 100.0
    
    f = 1.0 / np.tan(fov / 2.0)
    projmatrix = torch.tensor([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], device=device, dtype=torch.float32)
    
    campos = torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float32)
    
    return viewmatrix, projmatrix, campos


def test_basic_rendering():
    """Test basic rendering functionality"""
    print("\n" + "="*60)
    print("Test 1: Basic Rendering")
    print("="*60)
    
    device = 'cpu'
    width, height = 512, 512
    num_gaussians = 1000
    
    # Create test data
    print(f"Creating {num_gaussians} test Gaussians...")
    means3D, means2D, colors, opacities, scales, rotations = \
        create_test_gaussians(num_gaussians, device)
    
    viewmatrix, projmatrix, campos = create_camera_matrices(device)
    
    # Setup rasterization settings
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    tanfovx = tanfovy = 0.5
    
    raster_settings = idgr.GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False
    )
    
    # Create rasterizer
    rasterizer = idgr.GaussianRasterizer(raster_settings)
    
    # Render
    print("Rendering...")
    start_time = time.time()
    
    try:
        with torch.no_grad():
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                opacities=opacities,
                colors_precomp=colors,
                scales=scales,
                rotations=rotations
            )
        
        elapsed = time.time() - start_time
        
        print(f"✓ Rendering successful!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Output shape: {rendered_image.shape}")
        print(f"  Output range: [{rendered_image.min():.3f}, {rendered_image.max():.3f}]")
        print(f"  Radii: {(radii > 0).sum()}/{len(radii)} visible")
        
        return True
        
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mark_visible():
    """Test frustum culling"""
    print("\n" + "="*60)
    print("Test 2: Frustum Culling (markVisible)")
    print("="*60)
    
    device = 'cpu'
    num_gaussians = 1000
    
    means3D, _, _, _, _, _ = create_test_gaussians(num_gaussians, device)
    viewmatrix, projmatrix, _ = create_camera_matrices(device)
    
    raster_settings = idgr.GaussianRasterizationSettings(
        image_height=512,
        image_width=512,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=torch.zeros(3, device=device, dtype=torch.float32),
        prefiltered=False,
        debug=False
    )
    
    rasterizer = idgr.GaussianRasterizer(raster_settings)
    
    try:
        visible = rasterizer.markVisible(means3D)
        num_visible = visible.sum().item()
        
        print(f"✓ Frustum culling successful!")
        print(f"  Visible Gaussians: {num_visible}/{num_gaussians}")
        
        return True
        
    except Exception as e:
        print(f"✗ Frustum culling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spherical_harmonics():
    """Test rendering with spherical harmonics"""
    print("\n" + "="*60)
    print("Test 3: Spherical Harmonics Rendering")
    print("="*60)
    
    device = 'cpu'
    width, height = 512, 512
    num_gaussians = 500
    sh_degree = 3
    
    # Create test data
    means3D, means2D, _, opacities, scales, rotations = \
        create_test_gaussians(num_gaussians, device)
    
    # Create random SH coefficients
    # For degree 3: (degree+1)^2 = 16 coefficients per channel
    num_sh_coeffs = (sh_degree + 1) ** 2
    shs = torch.randn(num_gaussians, num_sh_coeffs, 3, device=device, dtype=torch.float32) * 0.1
    
    viewmatrix, projmatrix, campos = create_camera_matrices(device)
    
    raster_settings = idgr.GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = idgr.GaussianRasterizer(raster_settings)
    
    try:
        with torch.no_grad():
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                opacities=opacities,
                shs=shs,
                scales=scales,
                rotations=rotations
            )
        
        print(f"✓ SH rendering successful!")
        print(f"  SH degree: {sh_degree}")
        print(f"  SH coefficients: {num_sh_coeffs}")
        print(f"  Output shape: {rendered_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ SH rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark rendering performance"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)
    
    device = 'cpu'
    width, height = 800, 800
    
    test_configs = [
        (100, "100 Gaussians"),
        (1000, "1K Gaussians"),
        (10000, "10K Gaussians"),
        (50000, "50K Gaussians"),
        (100000, "100K Gaussians"),
        (150000, "150K Gaussians"),
        (200000, "200K Gaussians"),
        (500000, "500K Gaussians"),
    ]
    
    viewmatrix, projmatrix, campos = create_camera_matrices(device)
    
    print(f"\nResolution: {width}x{height}")
    print("-" * 60)
    
    for num_gaussians, label in test_configs:
        means3D, means2D, colors, opacities, scales, rotations = \
            create_test_gaussians(num_gaussians, device)
        
        raster_settings = idgr.GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=0.5,
            tanfovy=0.5,
            bg=torch.zeros(3, device=device, dtype=torch.float32),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False
        )
        
        rasterizer = idgr.GaussianRasterizer(raster_settings)
        
        # Warmup
        with torch.no_grad():
            _ = rasterizer(means3D, means2D, opacities, 
                          colors_precomp=colors, scales=scales, rotations=rotations)
        
        # Benchmark
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = rasterizer(means3D, means2D, opacities,
                              colors_precomp=colors, scales=scales, rotations=rotations)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_iterations
        fps = 1.0 / avg_time
        
        print(f"{label:20s}: {avg_time*1000:6.2f}ms/frame ({fps:5.2f} FPS)")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Intel XPU Gaussian Rasterization Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Rendering", test_basic_rendering()))
    results.append(("Frustum Culling", test_mark_visible()))
    results.append(("Spherical Harmonics", test_spherical_harmonics()))
    
    # Performance benchmark
    try:
        benchmark_performance()
    except Exception as e:
        print(f"\n⚠ Benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s}: {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
