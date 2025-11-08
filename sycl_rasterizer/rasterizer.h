#ifndef SYCL_RASTERIZER_H_INCLUDED
#define SYCL_RASTERIZER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

// Mark visible Gaussians (frustum culling)
void sycl_rasterizer_markVisible(
    int P,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    int* present);

// Forward pass
int sycl_rasterizer_forward(
    int P, int D, int M,
    const float* background,
    int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    float tan_fovx, float tan_fovy,
    bool prefiltered,
    float* out_color,
    int* radii,
    bool debug);

#ifdef __cplusplus
}
#endif

#endif
