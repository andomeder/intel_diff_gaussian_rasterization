#include <sycl/sycl.hpp>
#include "forward.h"
#include "auxiliary.h"
#include <cmath>

namespace sycl_rasterizer
{

    // Spherical Harmonics constants
    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_C1 = 0.4886025119029199f;
    constexpr float SH_C2[] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f};
    constexpr float SH_C3[] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f};

    // Compute color from Spherical Harmonics
    inline float3 computeColorFromSH(
        int idx,
        int deg,
        int max_coeffs,
        const float3 *means,
        const float3 &campos,
        const float *shs,
        int *clamped)
    {
        float3 pos = means[idx];
        float3 dir = pos - campos;
        float length = sycl::length(dir);
        dir = dir / length;

        const float3 *sh = (const float3 *)(shs + idx * max_coeffs * 3);
        float3 result = sh[0] * SH_C0;

        if (deg > 0)
        {
            float x = dir.x();
            float y = dir.y();
            float z = dir.z();

            result -= sh[1] * (SH_C1 * y);
            result += sh[2] * (SH_C1 * z);
            result -= sh[3] * (SH_C1 * x);

            if (deg > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;

                result += sh[4] * (SH_C2[0] * xy);
                result += sh[5] * (SH_C2[1] * yz);
                result += sh[6] * (SH_C2[2] * (2.0f * zz - xx - yy));
                result += sh[7] * (SH_C2[3] * xz);
                result += sh[8] * (SH_C2[4] * (xx - yy));

                if (deg > 2)
                {
                    result += sh[9] * (SH_C3[0] * y * (3.0f * xx - yy));
                    result += sh[10] * (SH_C3[1] * xy * z);
                    result += sh[11] * (SH_C3[2] * y * (4.0f * zz - xx - yy));
                    result += sh[12] * (SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy));
                    result += sh[13] * (SH_C3[4] * x * (4.0f * zz - xx - yy));
                    result += sh[14] * (SH_C3[5] * z * (xx - yy));
                    result += sh[15] * (SH_C3[6] * x * (xx - 3.0f * yy));
                }
            }
        }

        result += float3(0.5f, 0.5f, 0.5f);

        // Record clamping
        clamped[3 * idx + 0] = (result.x() < 0.0f) ? 1 : 0;
        clamped[3 * idx + 1] = (result.y() < 0.0f) ? 1 : 0;
        clamped[3 * idx + 2] = (result.z() < 0.0f) ? 1 : 0;

        result = sycl::max(result, float3(0.0f, 0.0f, 0.0f));
        return result;
    }

    // Compute 2D covariance from 3D covariance
    inline float3 computeCov2D(
        const float3 &mean,
        float focal_x,
        float focal_y,
        float tan_fovx,
        float tan_fovy,
        const float *cov3D,
        const float *viewmatrix)
    {
        float3 t = transformPoint4x3(mean, viewmatrix);

        float limx = 1.3f * tan_fovx;
        float limy = 1.3f * tan_fovy;
        float txtz = t.x() / t.z();
        float tytz = t.y() / t.z();
        t.x() = sycl::min(limx, sycl::max(-limx, txtz)) * t.z();
        t.y() = sycl::min(limy, sycl::max(-limy, tytz)) * t.z();

        // Jacobian of perspective projection
        float J[9] = {
            focal_x / t.z(), 0.0f, -(focal_x * t.x()) / (t.z() * t.z()),
            0.0f, focal_y / t.z(), -(focal_y * t.y()) / (t.z() * t.z()),
            0.0f, 0.0f, 0.0f};

        // View matrix (rotation part)
        float W[9] = {
            viewmatrix[0], viewmatrix[4], viewmatrix[8],
            viewmatrix[1], viewmatrix[5], viewmatrix[9],
            viewmatrix[2], viewmatrix[6], viewmatrix[10]};

        // T = W * J
        float T[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                T[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; k++)
                {
                    T[i * 3 + j] += W[i * 3 + k] * J[k * 3 + j];
                }
            }
        }

        // Reconstruct symmetric 3D covariance matrix
        float Vrk[9] = {
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]};

        // Compute cov = T^T * Vrk * T
        float temp[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                temp[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; k++)
                {
                    temp[i * 3 + j] += T[k * 3 + i] * Vrk[k * 3 + j];
                }
            }
        }

        float cov[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cov[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; k++)
                {
                    cov[i * 3 + j] += temp[i * 3 + k] * T[k * 3 + j];
                }
            }
        }

        // Add low-pass filter
        cov[0] += 0.3f;
        cov[4] += 0.3f;

        // Return upper-triangular elements of 2D covariance
        return float3(cov[0], cov[1], cov[4]);
    }

    // Compute 3D covariance from scale and rotation
    inline void computeCov3D(
        const float3 &scale,
        float mod,
        const float4 &rot,
        float *cov3D)
    {
        // Scaling matrix
        float S[9] = {
            mod * scale.x(), 0.0f, 0.0f,
            0.0f, mod * scale.y(), 0.0f,
            0.0f, 0.0f, mod * scale.z()};

        // Rotation matrix from quaternion
        float qx = rot.x();
        float qy = rot.y();
        float qz = rot.z();
        float qw = rot.w();

        float R[9] = {
            1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - qw * qz), 2.f * (qx * qz + qw * qy),
            2.f * (qx * qy + qw * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - qw * qx),
            2.f * (qx * qz - qw * qy), 2.f * (qy * qz + qw * qx), 1.f - 2.f * (qx * qx + qy * qy)};

        // M = S * R
        float M[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                M[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; k++)
                {
                    M[i * 3 + j] += S[i * 3 + k] * R[k * 3 + j];
                }
            }
        }

        // Sigma = M^T * M
        float Sigma[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Sigma[i * 3 + j] = 0.0f;
                for (int k = 0; k < 3; k++)
                {
                    Sigma[i * 3 + j] += M[k * 3 + i] * M[k * 3 + j];
                }
            }
        }

        // Store upper-triangular part
        cov3D[0] = Sigma[0];
        cov3D[1] = Sigma[1];
        cov3D[2] = Sigma[2];
        cov3D[3] = Sigma[4];
        cov3D[4] = Sigma[5];
        cov3D[5] = Sigma[8];
    }

    // SYCL Kernel: Preprocessing
    class PreprocessKernel;

    void preprocessSYCL(
        sycl::queue &q,
        int P, int D, int M,
        const float *orig_points,
        // const float3* scales,
        const float *scales,
        const float scale_modifier,
        // const float4* rotations,
        const float *rotations,
        const float *opacities,
        const float *shs,
        int *clamped,
        const float *cov3D_precomp,
        const float *colors_precomp,
        const float *viewmatrix,
        const float *projmatrix,
        const float3 &cam_pos,
        const int W, int H,
        const float tan_fovx, float tan_fovy,
        const float focal_x, float focal_y,
        int *radii,
        float2 *points_xy_image,
        float *depths,
        float *cov3Ds,
        float *rgb,
        float4 *conic_opacity,
        const dim3 &grid,
        uint32_t *tiles_touched,
        bool prefiltered)
    {

    // Launch kernel with one work-item per Gaussian
    auto event = q.parallel_for<PreprocessKernel>(
        sycl::range<1>(P),
        [=](sycl::id<1> idx) {

            int i = idx[0];

            // Initialize outputs
            radii[i] = 0;
            tiles_touched[i] = 0;

            // Frustum culling
            float3 p_view;
            if (!inFrustum(i, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) {
                return;
            }

            // Transform to clip space
            float3 p_orig(orig_points[3 * i], orig_points[3 * i + 1], orig_points[3 * i + 2]);
            float4 p_hom = transformPoint4x4(p_orig, projmatrix);
            float p_w = 1.0f / (p_hom.w() + 1e-7f);
            float3 p_proj(p_hom.x() * p_w, p_hom.y() * p_w, p_hom.z() * p_w);

            // Get 3D covariance
            const float* cov3D;
            if (cov3D_precomp != nullptr) {
                cov3D = cov3D_precomp + i * 6;
            } else {
                // Reconstruct vectors from the flat float arrays
                float3 scale_vec(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
                float4 rot_vec(rotations[i * 4 + 0], rotations[i * 4 + 1], rotations[i * 4 + 2], rotations[i * 4 + 3]);

                //computeCov3D(scales[i], scale_modifier, rotations[i], cov3Ds + i * 6);
                computeCov3D(scale_vec, scale_modifier, rot_vec, cov3Ds + i * 6);
                cov3D = cov3Ds + i * 6;
            }

            // Compute 2D covariance
            float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

            // Compute inverse covariance (conic)
            float det = (cov.x() * cov.z() - cov.y() * cov.y());
            if (det == 0.0f) return;
            
            float det_inv = 1.0f / det;
            float3 conic(cov.z() * det_inv, -cov.y() * det_inv, cov.x() * det_inv);

            // Compute radius from eigenvalues
            float mid = 0.5f * (cov.x() + cov.z());
            float delta = mid * mid - det;
            delta = sycl::max(delta, 0.1f);
            float sqrt_delta = sycl::sqrt(delta);
            float lambda1 = mid + sqrt_delta;
            float lambda2 = mid - sqrt_delta;
            float my_radius = sycl::ceil(3.0f * sycl::sqrt(sycl::max(lambda1, lambda2)));

            // Project to screen space
            float2 point_image(ndc2Pix(p_proj.x(), W), ndc2Pix(p_proj.y(), H));

            // Get tile bounds
            uint2 rect_min, rect_max;
            getRect(point_image, (int)my_radius, rect_min, rect_max, grid);
            
            if ((rect_max.x() - rect_min.x()) * (rect_max.y() - rect_min.y()) == 0) {
                return;
            }

            // Compute color from SH if needed
            if (colors_precomp == nullptr) {
                float3 result = computeColorFromSH(i, D, M, (const float3*)orig_points, cam_pos, shs, clamped);
                rgb[i * NUM_CHANNELS + 0] = result.x();
                rgb[i * NUM_CHANNELS + 1] = result.y();
                rgb[i * NUM_CHANNELS + 2] = result.z();
            }

            // Store results
            depths[i] = p_view.z();
            radii[i] = (int)my_radius;
            points_xy_image[i] = point_image;
            conic_opacity[i] = float4(conic.x(), conic.y(), conic.z(), opacities[i]);
            tiles_touched[i] = (rect_max.y() - rect_min.y()) * (rect_max.x() - rect_min.x());
        }
    );

        event.wait();
    }
}
