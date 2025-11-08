#ifndef SYCL_RASTERIZER_AUXILIARY_H_INCLUDED
#define SYCL_RASTERIZER_AUXILIARY_H_INCLUDED

#include <sycl/sycl.hpp>
#include "config.h"

namespace sycl_rasterizer {

// Vector types using SYCL built-ins
using float2 = sycl::float2;
using float3 = sycl::float3;
using float4 = sycl::float4;
using uint2 = sycl::uint2;
using uint3 = sycl::uint3;
using uint4 = sycl::uint4;

// dim3 equivalent for SYCL
struct dim3 {
    uint32_t x, y, z;
    dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Get rectangle bounds for a point with radius
inline void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, const dim3& grid) {
    rect_min.x() = sycl::min((uint32_t)grid.x, sycl::max((uint32_t)0, (uint32_t)((p.x() - max_radius) / BLOCK_X)));
    rect_min.y() = sycl::min((uint32_t)grid.y, sycl::max((uint32_t)0, (uint32_t)((p.y() - max_radius) / BLOCK_Y)));
    rect_max.x() = sycl::min((uint32_t)grid.x, sycl::max((uint32_t)0, (uint32_t)((p.x() + max_radius + BLOCK_X - 1) / BLOCK_X)));
    rect_max.y() = sycl::min((uint32_t)grid.y, sycl::max((uint32_t)0, (uint32_t)((p.y() + max_radius + BLOCK_Y - 1) / BLOCK_Y)));
}

// 4x3 matrix transform (for view transform)
inline float3 transformPoint4x3(const float3& p, const float* matrix) {
    return float3(
        matrix[0] * p.x() + matrix[4] * p.y() + matrix[8]  * p.z() + matrix[12],
        matrix[1] * p.x() + matrix[5] * p.y() + matrix[9]  * p.z() + matrix[13],
        matrix[2] * p.x() + matrix[6] * p.y() + matrix[10] * p.z() + matrix[14]
    );
}

// 4x4 matrix transform (for projection)
inline float4 transformPoint4x4(const float3& p, const float* matrix) {
    return float4(
        matrix[0] * p.x() + matrix[4] * p.y() + matrix[8]  * p.z() + matrix[12],
        matrix[1] * p.x() + matrix[5] * p.y() + matrix[9]  * p.z() + matrix[13],
        matrix[2] * p.x() + matrix[6] * p.y() + matrix[10] * p.z() + matrix[14],
        matrix[3] * p.x() + matrix[7] * p.y() + matrix[11] * p.z() + matrix[15]
    );
}

// Frustum culling check
inline bool inFrustum(
    int idx,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool prefiltered,
    float3& p_view)
{
    float3 p_orig(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
    
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w() + 1e-7f);
    float3 p_proj(p_hom.x() * p_w, p_hom.y() * p_w, p_hom.z() * p_w);
    
    p_view = transformPoint4x3(p_orig, viewmatrix);
    
    if (p_view.z() <= 0.2f) {
        return false;
    }
    return true;
}

// NDC to pixel coordinates
inline float ndc2Pix(float v, int S) {
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

// Compute higher MSB for radix sort
inline uint32_t getHigherMsb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

}

#endif
