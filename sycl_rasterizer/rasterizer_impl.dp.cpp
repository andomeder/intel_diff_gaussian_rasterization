#include <sycl/sycl.hpp>
// #include <dpct/dpct.hpp>
#include "rasterizer.h"
#include "forward.h"
#include "auxiliary.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cstring>

namespace sycl_rasterizer
{

    // Global queue - initialized on first use
    static sycl::queue *g_queue = nullptr;

    void cleanup_sycl_queue() {
        if (g_queue) {
            delete g_queue;
            g_queue = nullptr;
        }
    }

    sycl::queue& get_queue() {
    if (!g_queue) {
        std::atexit(cleanup_sycl_queue);
        try {
            sycl::queue q(sycl::gpu_selector_v);

            std::cout << "SUCCESS: Automatically selected GPU:" << std::endl;
            std::cout << "  Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
            std::cout << "  Backend: " << q.get_device().get_backend() << std::endl;

            g_queue = new sycl::queue(q);
            //return *g_queue;

        } catch (sycl::exception const& e) {
            std::cerr << "FATAL ERROR: GPU selection failed: " << e.what() << std::endl;
            throw std::runtime_error("Could not initialize a SYCL GPU queue.");
        }
    }
    return *g_queue;
}

    // Kernel: Duplicate Gaussians with keys for sorting
    class DuplicateWithKeysKernel;

    void duplicateWithKeysSYCL(
        sycl::queue &q,
        int P,
        const float2 *points_xy,
        const float *depths,
        const uint32_t *offsets,
        uint64_t *gaussian_keys_unsorted,
        uint32_t *gaussian_values_unsorted,
        const int *radii,
        const dim3 &grid)
    {
        auto event = q.parallel_for<DuplicateWithKeysKernel>(
            sycl::range<1>(P),
            [=](sycl::id<1> idx)
            {
                int i = idx[0];

                if (radii[i] > 0)
                {
                    uint32_t off = (i == 0) ? 0 : offsets[i - 1];

                    uint2 rect_min, rect_max;
                    getRect(points_xy[i], radii[i], rect_min, rect_max, grid);

                    for (uint32_t y = rect_min.y(); y < rect_max.y(); y++)
                    {
                        for (uint32_t x = rect_min.x(); x < rect_max.x(); x++)
                        {
                            uint64_t key = (uint64_t)(y * grid.x + x);
                            key <<= 32;

                            // Pack depth as uint32_t into lower 32 bits
                            uint32_t depth_as_uint;
                            // std::memcpy(&depth_as_uint, &depths[i], sizeof(float));
                            depth_as_uint = sycl::bit_cast<uint32_t>(depths[i]);
                            key |= depth_as_uint;

                            gaussian_keys_unsorted[off] = key;
                            gaussian_values_unsorted[off] = i;
                            off++;
                        }
                    }
                }
            });

        event.wait();
    }

    // Kernel: Identify tile ranges after sorting
    class IdentifyTileRangesKernel;

    void identifyTileRangesSYCL(
        sycl::queue &q,
        int L,
        const uint64_t *point_list_keys,
        uint2 *ranges)
    {
        auto event = q.parallel_for<IdentifyTileRangesKernel>(
            sycl::range<1>(L),
            [=](sycl::id<1> idx)
            {
                int i = idx[0];

                uint64_t key = point_list_keys[i];
                uint32_t currtile = (uint32_t)(key >> 32);

                if (i == 0)
                {
                    ranges[currtile].x() = 0;
                }
                else
                {
                    uint32_t prevtile = (uint32_t)(point_list_keys[i - 1] >> 32);
                    if (currtile != prevtile)
                    {
                        ranges[prevtile].y() = i;
                        ranges[currtile].x() = i;
                    }
                }

                if (i == L - 1)
                {
                    ranges[currtile].y() = L;
                }
            });

        event.wait();
    }

    // Radix sort implementation for SYCL
    void radixSortSYCL(
        sycl::queue &q,
        int num_items,
        uint64_t *keys_in,
        uint64_t *keys_out,
        uint32_t *values_in,
        uint32_t *values_out)
    {
        // TODO: optimize with parallel radix sort later
        std::vector<uint64_t> keys_host(num_items);
        std::vector<uint32_t> values_host(num_items);

        q.memcpy(keys_host.data(), keys_in, num_items * sizeof(uint64_t)).wait();
        q.memcpy(values_host.data(), values_in, num_items * sizeof(uint32_t)).wait();

        // Create indices and sort
        std::vector<int> indices(num_items);
        for (int i = 0; i < num_items; i++)
        {
            indices[i] = i;
        }

        std::sort(indices.begin(), indices.end(),
                  [&keys_host](int a, int b)
                  {
                      return keys_host[a] < keys_host[b];
                  });

        // Reorder based on sorted indices
        std::vector<uint64_t> sorted_keys_host(num_items);
        std::vector<uint32_t> sorted_values_host(num_items);
        for (int i = 0; i < num_items; i++)
        {
            sorted_keys_host[i] = keys_host[indices[i]];
            sorted_values_host[i] = values_host[indices[i]];
        }

        q.memcpy(keys_out, sorted_keys_host.data(), num_items * sizeof(uint64_t)).wait();
        q.memcpy(values_out, sorted_values_host.data(), num_items * sizeof(uint32_t)).wait();
    }

    // Main forward pass
    int forward(
        int P, int D, int M,
        const float *background,
        int width, int height,
        const float *means3D,
        const float *shs,
        const float *colors_precomp,
        const float *opacities,
        const float *scales,
        float scale_modifier,
        const float *rotations,
        const float *cov3D_precomp,
        const float *viewmatrix,
        const float *projmatrix,
        const float *cam_pos,
        float tan_fovx, float tan_fovy,
        bool prefiltered,
        float *out_color,
        int *radii,
        bool debug)
    {
        sycl::queue &q = get_queue();

        const float3 h_cam_pos(cam_pos[0], cam_pos[1], cam_pos[2]);
        const float focal_y = height / (2.0f * tan_fovy);
        const float focal_x = width / (2.0f * tan_fovx);

        dim3 tile_grid;
        tile_grid.x = (width + BLOCK_X - 1) / BLOCK_X;
        tile_grid.y = (height + BLOCK_Y - 1) / BLOCK_Y;
        tile_grid.z = 1;

        // Allocate device memory using USM
        float *d_depths = sycl::malloc_device<float>(P, q);
        int *d_clamped = sycl::malloc_device<int>(P * 3, q);
        int *d_internal_radii = sycl::malloc_device<int>(P, q);
        float2 *d_means2D = sycl::malloc_device<float2>(P, q);
        float *d_cov3D = sycl::malloc_device<float>(P * 6, q);
        float4 *d_conic_opacity = sycl::malloc_device<float4>(P, q);
        float *d_rgb = sycl::malloc_device<float>(P * NUM_CHANNELS, q);
        uint32_t *d_tiles_touched = sycl::malloc_device<uint32_t>(P, q);
        uint32_t *d_point_offsets = sycl::malloc_device<uint32_t>(P, q);

        // Copy input data to device
        float *d_means3D = sycl::malloc_device<float>(P * 3, q);
        //float3 *d_scales = sycl::malloc_device<float3>(P, q);
        //float4 *d_rotations = sycl::malloc_device<float4>(P, q);
        float* d_scales = sycl::malloc_device<float>(P * 3, q);
        float* d_rotations = sycl::malloc_device<float>(P * 4, q);
        float *d_opacities = sycl::malloc_device<float>(P, q);
        float *d_shs = nullptr;
        float *d_colors_precomp = nullptr;
        float *d_cov3D_precomp = nullptr;
        float *d_viewmatrix = sycl::malloc_device<float>(16, q);
        float *d_projmatrix = sycl::malloc_device<float>(16, q);
        // float3* d_cam_pos = sycl::malloc_device<float3>(1, q);

        q.memcpy(d_means3D, means3D, P * 3 * sizeof(float));
        //q.memcpy(d_scales, scales, P * sizeof(float3));
        //q.memcpy(d_rotations, rotations, P * sizeof(float4));
        q.memcpy(d_scales, scales, P * 3 * sizeof(float));
        q.memcpy(d_rotations, rotations, P * 4 * sizeof(float));
        q.memcpy(d_opacities, opacities, P * sizeof(float));
        q.memcpy(d_viewmatrix, viewmatrix, 16 * sizeof(float));
        q.memcpy(d_projmatrix, projmatrix, 16 * sizeof(float));
        // q.memcpy(d_cam_pos, cam_pos, sizeof(float3));

        if (shs != nullptr && M > 0)
        {
            d_shs = sycl::malloc_device<float>(P * M * 3, q);
            q.memcpy(d_shs, shs, P * M * 3 * sizeof(float));
        }

        if (colors_precomp != nullptr)
        {
            d_colors_precomp = sycl::malloc_device<float>(P * NUM_CHANNELS, q);
            q.memcpy(d_colors_precomp, colors_precomp, P * NUM_CHANNELS * sizeof(float));
        }

        if (cov3D_precomp != nullptr)
        {
            d_cov3D_precomp = sycl::malloc_device<float>(P * 6, q);
            q.memcpy(d_cov3D_precomp, cov3D_precomp, P * 6 * sizeof(float));
        }

        q.wait();

        //int *d_radii = (radii == nullptr) ? d_internal_radii : radii;
        int* d_radii = d_internal_radii;

        // Preprocessing
        preprocessSYCL(
            q, P, D, M,
            d_means3D,
            d_scales,
            scale_modifier,
            d_rotations,
            d_opacities,
            d_shs,
            d_clamped,
            d_cov3D_precomp,
            d_colors_precomp,
            d_viewmatrix,
            d_projmatrix,
            //*d_cam_pos,
            h_cam_pos,
            width, height,
            tan_fovx, tan_fovy,
            focal_x, focal_y,
            d_radii,
            d_means2D,
            d_depths,
            d_cov3D,
            d_rgb,
            d_conic_opacity,
            tile_grid,
            d_tiles_touched,
            prefiltered);

        // Compute prefix sum of tiles_touched
        std::vector<uint32_t> tiles_touched_host(P);
        q.memcpy(tiles_touched_host.data(), d_tiles_touched, P * sizeof(uint32_t)).wait();

        // Perform an exclusive scan to get the starting offset for each Gaussian.
        std::vector<uint32_t> point_offsets_host(P);
        uint32_t current_offset = 0;
        for (int i = 0; i < P; ++i)
        {
            point_offsets_host[i] = current_offset;
            current_offset += tiles_touched_host[i];
        }
        uint32_t total_rendered = current_offset;

        q.memcpy(d_point_offsets, point_offsets_host.data(), P * sizeof(uint32_t)).wait();

        int num_rendered = total_rendered;

        if (num_rendered == 0)
        {
            // No Gaussians visible, just return background
            for (int i = 0; i < width * height * NUM_CHANNELS; i++)
            {
                out_color[i] = background[i % NUM_CHANNELS];
            }

            // Cleanup
            sycl::free(d_depths, q);
            sycl::free(d_clamped, q);
            sycl::free(d_internal_radii, q);
            sycl::free(d_means2D, q);
            sycl::free(d_cov3D, q);
            sycl::free(d_conic_opacity, q);
            sycl::free(d_rgb, q);
            sycl::free(d_tiles_touched, q);
            sycl::free(d_point_offsets, q);
            sycl::free(d_means3D, q);
            sycl::free(d_scales, q);
            sycl::free(d_rotations, q);
            sycl::free(d_opacities, q);
            if (d_shs)
                sycl::free(d_shs, q);
            if (d_colors_precomp)
                sycl::free(d_colors_precomp, q);
            if (d_cov3D_precomp)
                sycl::free(d_cov3D_precomp, q);
            sycl::free(d_viewmatrix, q);
            sycl::free(d_projmatrix, q);
            // sycl::free(d_cam_pos, q);

            return 0;
        }

        // Allocate sorting buffers
        uint32_t *d_point_list = sycl::malloc_device<uint32_t>(num_rendered, q);
        uint32_t *d_point_list_unsorted = sycl::malloc_device<uint32_t>(num_rendered, q);
        uint64_t *d_point_list_keys = sycl::malloc_device<uint64_t>(num_rendered, q);
        uint64_t *d_point_list_keys_unsorted = sycl::malloc_device<uint64_t>(num_rendered, q);

        // Duplicate with keys
        duplicateWithKeysSYCL(
            q, P,
            d_means2D,
            d_depths,
            d_point_offsets,
            d_point_list_keys_unsorted,
            d_point_list_unsorted,
            d_radii,
            tile_grid);

        // Sort by keys
        radixSortSYCL(
            q, num_rendered,
            d_point_list_keys_unsorted,
            d_point_list_keys,
            d_point_list_unsorted,
            d_point_list);

        // Allocate image state
        int num_tiles = tile_grid.x * tile_grid.y;
        uint2 *d_ranges = sycl::malloc_device<uint2>(num_tiles, q);
        uint32_t *d_n_contrib = sycl::malloc_device<uint32_t>(width * height, q);
        float *d_accum_alpha = sycl::malloc_device<float>(width * height, q);
        float *d_out_color = sycl::malloc_device<float>(NUM_CHANNELS * width * height, q);
        float *d_bg_color = sycl::malloc_device<float>(NUM_CHANNELS, q);

        // Initialize ranges to zero
        q.memset(d_ranges, 0, num_tiles * sizeof(uint2)).wait();
        q.memcpy(d_bg_color, background, NUM_CHANNELS * sizeof(float)).wait();

        // Identify tile ranges
        identifyTileRangesSYCL(q, num_rendered, d_point_list_keys, d_ranges);

        // Render
        const float *feature_ptr = (colors_precomp != nullptr) ? d_colors_precomp : d_rgb;

        renderSYCLOptimized(
            q,
            tile_grid,
            d_ranges,
            d_point_list,
            width, height,
            d_means2D,
            feature_ptr,
            d_conic_opacity,
            d_accum_alpha,
            d_n_contrib,
            d_bg_color,
            d_out_color);

        // Copy result back to host
        q.memcpy(out_color, d_out_color, NUM_CHANNELS * width * height * sizeof(float)).wait();

        if (radii != nullptr) {
            q.memcpy(radii, d_radii, P * sizeof(int)).wait();
        }

        // Cleanup
        sycl::free(d_depths, q);
        sycl::free(d_clamped, q);
        sycl::free(d_internal_radii, q);
        sycl::free(d_means2D, q);
        sycl::free(d_cov3D, q);
        sycl::free(d_conic_opacity, q);
        sycl::free(d_rgb, q);
        sycl::free(d_tiles_touched, q);
        sycl::free(d_point_offsets, q);
        sycl::free(d_means3D, q);
        sycl::free(d_scales, q);
        sycl::free(d_rotations, q);
        sycl::free(d_opacities, q);
        if (d_shs)
            sycl::free(d_shs, q);
        if (d_colors_precomp)
            sycl::free(d_colors_precomp, q);
        if (d_cov3D_precomp)
            sycl::free(d_cov3D_precomp, q);
        sycl::free(d_viewmatrix, q);
        sycl::free(d_projmatrix, q);
        // sycl::free(d_cam_pos, q);
        sycl::free(d_point_list, q);
        sycl::free(d_point_list_unsorted, q);
        sycl::free(d_point_list_keys, q);
        sycl::free(d_point_list_keys_unsorted, q);
        sycl::free(d_ranges, q);
        sycl::free(d_n_contrib, q);
        sycl::free(d_accum_alpha, q);
        sycl::free(d_out_color, q);
        sycl::free(d_bg_color, q);

        return num_rendered;
    }

}

// C API wrapper
extern "C"
{

    int sycl_rasterizer_forward(
        int P, int D, int M,
        const float *background,
        int width, int height,
        const float *means3D,
        const float *shs,
        const float *colors_precomp,
        const float *opacities,
        const float *scales,
        float scale_modifier,
        const float *rotations,
        const float *cov3D_precomp,
        const float *viewmatrix,
        const float *projmatrix,
        const float *cam_pos,
        float tan_fovx, float tan_fovy,
        bool prefiltered,
        float *out_color,
        int *radii,
        bool debug)
    {
        return sycl_rasterizer::forward(
            P, D, M,
            background,
            width, height,
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            scale_modifier,
            rotations,
            cov3D_precomp,
            viewmatrix,
            projmatrix,
            cam_pos,
            tan_fovx, tan_fovy,
            prefiltered,
            out_color,
            radii,
            debug);
    }

    void sycl_rasterizer_markVisible(
        int P,
        float *means3D,
        float *viewmatrix,
        float *projmatrix,
        int *present)
    {
        // Simple CPU implementation for mark visible
        for (int i = 0; i < P; i++)
        {
            sycl_rasterizer::float3 p_view;
            present[i] = sycl_rasterizer::inFrustum(i, means3D, viewmatrix, projmatrix, false, p_view) ? 1 : 0;
        }
    }

}
