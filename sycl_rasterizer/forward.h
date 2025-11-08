#ifndef SYCL_RASTERIZER_FORWARD_H_INCLUDED
#define SYCL_RASTERIZER_FORWARD_H_INCLUDED

#include <sycl/sycl.hpp>
#include "auxiliary.h"

namespace sycl_rasterizer {

// Preprocessing kernel
void preprocessSYCL(
    sycl::queue& q,
    int P, int D, int M,
    const float* orig_points,
    //const float3* scales,
    const float* scales,
    const float scale_modifier,
    //const float4* rotations,
    const float* rotations,
    const float* opacities,
    const float* shs,
    int* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float3& cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3& grid,
    uint32_t* tiles_touched,
    bool prefiltered);

// Rendering kernel (basic version)
void renderSYCL(
    sycl::queue& q,
    const dim3& grid,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* points_xy_image,
    const float* features,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color);

// Rendering kernel (optimized with shared memory)
void renderSYCLOptimized(
    sycl::queue& q,
    const dim3& grid,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* points_xy_image,
    const float* features,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color);

}

#endif
