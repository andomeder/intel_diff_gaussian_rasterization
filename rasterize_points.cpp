/*
 * Intel XPU Gaussian Rasterization PyTorch Extension
 */

#include <torch/extension.h>
#include <iostream>
#include <tuple>
#include <vector>

extern "C" {
#include "sycl_rasterizer/rasterizer.h"
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansIntel(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug)
{
    // Validate inputs
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    
    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    // Create output tensors
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt32);
    
    torch::Tensor out_color = torch::zeros({3, H, W}, float_opts);
    torch::Tensor radii = torch::zeros({P}, int_opts);
    
    // Placeholder buffers (not used in forward-only implementation)
    torch::Device device(torch::kCPU);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    
    int rendered = 0;
    
    if (P != 0) {
        int M = 0;
        if (sh.size(0) != 0) {
            M = sh.size(1);
        }
        
        // Call SYCL rasterizer
        rendered = sycl_rasterizer_forward(
            P, degree, M,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh.size(0) > 0 ? sh.contiguous().data_ptr<float>() : nullptr,
            colors.size(0) > 0 ? colors.contiguous().data_ptr<float>() : nullptr,
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.size(0) > 0 ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            debug
        );
    }
    
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardIntel(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug) 
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);
    
    int M = 0;
    if (sh.size(0) != 0) {
        M = sh.size(1);
    }

    // Create gradient tensors
    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    
    // TODO: Implement backward pass
    // For now, return zero gradients
    std::cerr << "WARNING: Backward pass not yet implemented for Intel XPU version" << std::endl;
    
    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, 
                           dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{ 
    const int P = means3D.size(0);
    
    torch::Tensor present = torch::zeros({P}, means3D.options().dtype(torch::kInt32));
    
    if (P != 0) {
        sycl_rasterizer_markVisible(
            P,
            means3D.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            present.contiguous().data_ptr<int>()
        );
    }
    
    return present;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians", &RasterizeGaussiansIntel, 
          "Intel XPU Gaussian Rasterization (forward)");
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardIntel,
          "Intel XPU Gaussian Rasterization (backward)");
    m.def("mark_visible", &markVisible,
          "Mark visible Gaussians");
}
