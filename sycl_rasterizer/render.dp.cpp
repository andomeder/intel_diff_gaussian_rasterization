#include <sycl/sycl.hpp>
#include "forward.h"
#include "auxiliary.h"

namespace sycl_rasterizer {

// SYCL Kernel: Tile-based rendering with alpha blending
class RenderKernel;

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
    float* out_color)
{
    uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    
    // Launch one work-group per tile
    sycl::range<2> global_size(vertical_blocks * BLOCK_Y, horizontal_blocks * BLOCK_X);
    sycl::range<2> local_size(BLOCK_Y, BLOCK_X);
    
    auto event = q.submit([&](sycl::handler& cgh) {
        
        cgh.parallel_for<RenderKernel>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                // Get pixel coordinates
                uint32_t pix_x = item.get_global_id(1);
                uint32_t pix_y = item.get_global_id(0);
                
                // Get tile coordinates
                uint32_t tile_x = item.get_group(1);
                uint32_t tile_y = item.get_group(0);
                
                // Check if pixel is within image bounds
                if (pix_x >= (uint32_t)W || pix_y >= (uint32_t)H) return;
                
                uint32_t pix_id = W * pix_y + pix_x;
                float2 pixf(static_cast<float>(pix_x), static_cast<float>(pix_y));
                
                // Get range of Gaussians for this tile
                uint2 range = ranges[tile_y * horizontal_blocks + tile_x];
                
                // Initialize pixel state
                float T = 1.0f;
                uint32_t contributor = 0;
                uint32_t last_contributor = 0;
                float C[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f};
                
                // Iterate through Gaussians affecting this tile
                for (uint32_t idx = range.x(); idx < range.y(); idx++) {
                    contributor++;
                    uint32_t coll_id = point_list[idx];
                    
                    // Get Gaussian properties
                    float2 xy = points_xy_image[coll_id];
                    float2 d = xy - pixf;
                    float4 con_o = conic_opacity[coll_id];
                    
                    // Compute Gaussian weight
                    float power = -0.5f * (con_o.x() * d.x() * d.x() + 
                                           con_o.z() * d.y() * d.y()) - 
                                  con_o.y() * d.x() * d.y();
                    
                    if (power > 0.0f) continue;
                    
                    // Compute alpha
                    float alpha = sycl::min(0.99f, con_o.w() * sycl::exp(power));
                    if (alpha < 1.0f / 255.0f) continue;
                    
                    // Early ray termination
                    float test_T = T * (1.0f - alpha);
                    if (test_T < 0.0001f) {
                        break;
                    }
                    
                    // Alpha blending
                    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                        C[ch] += features[coll_id * NUM_CHANNELS + ch] * alpha * T;
                    }
                    
                    T = test_T;
                    last_contributor = contributor;
                }
                
                // Store final values
                final_T[pix_id] = T;
                n_contrib[pix_id] = last_contributor;
                
                // Composite with background
                for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                    out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
                }
            }
        );
    });
    
    event.wait();
}

// Optimized version using sub-groups for better memory access patterns
class RenderKernelOptimized;

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
    float* out_color)
{
    uint32_t vertical_blocks = (H + BLOCK_Y - 1) / BLOCK_Y;
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    
    sycl::range<2> global_size(vertical_blocks * BLOCK_Y, horizontal_blocks * BLOCK_X);
    sycl::range<2> local_size(BLOCK_Y, BLOCK_X);
    
    auto event = q.submit([&](sycl::handler& cgh) {
        // Shared memory for cooperative loading
        sycl::local_accessor<float2, 1> shared_positions(sycl::range<1>(256), cgh);
        sycl::local_accessor<float4, 1> shared_conic_opacity(sycl::range<1>(256), cgh);
        
        cgh.parallel_for<RenderKernelOptimized>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                uint32_t pix_x = item.get_global_id(1);
                uint32_t pix_y = item.get_global_id(0);
                
                uint32_t tile_x = item.get_group(1);
                uint32_t tile_y = item.get_group(0);
                
                uint32_t local_id = item.get_local_linear_id();
                
                if (pix_x >= (uint32_t)W || pix_y >= (uint32_t)H) return;
                
                uint32_t pix_id = W * pix_y + pix_x;
                float2 pixf(static_cast<float>(pix_x), static_cast<float>(pix_y));
                
                uint2 range = ranges[tile_y * horizontal_blocks + tile_x];
                
                float T = 1.0f;
                uint32_t contributor = 0;
                uint32_t last_contributor = 0;
                float C[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f};
                
                // Process Gaussians in batches
                for (uint32_t batch_start = range.x(); 
                     batch_start < range.y(); 
                     batch_start += BLOCK_SIZE) {
                    
                    uint32_t batch_size = sycl::min((uint32_t)BLOCK_SIZE, 
                                                     range.y() - batch_start);
                    
                    // Cooperative load into shared memory
                    if (local_id < batch_size) {
                        uint32_t gauss_id = point_list[batch_start + local_id];
                        shared_positions[local_id] = points_xy_image[gauss_id];
                        shared_conic_opacity[local_id] = conic_opacity[gauss_id];
                    }
                    
                    // Synchronize work-group
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Process batch
                    for (uint32_t i = 0; i < batch_size; i++) {
                        contributor++;
                        uint32_t coll_id = point_list[batch_start + i];
                        
                        float2 xy = shared_positions[i];
                        float2 d = xy - pixf;
                        float4 con_o = shared_conic_opacity[i];
                        
                        float power = -0.5f * (con_o.x() * d.x() * d.x() + 
                                               con_o.z() * d.y() * d.y()) - 
                                      con_o.y() * d.x() * d.y();
                        
                        if (power > 0.0f) continue;
                        
                        float alpha = sycl::min(0.99f, con_o.w() * sycl::exp(power));
                        if (alpha < 1.0f / 255.0f) continue;
                        
                        float test_T = T * (1.0f - alpha);
                        if (test_T < 0.0001f) {
                            contributor = range.y() - range.x() + 1; // Force break
                            break;
                        }
                        
                        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                            C[ch] += features[coll_id * NUM_CHANNELS + ch] * alpha * T;
                        }
                        
                        T = test_T;
                        last_contributor = contributor;
                    }
                    
                    // Synchronize before next batch
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Early termination if transmittance is too low
                    if (T < 0.0001f) break;
                }
                
                final_T[pix_id] = T;
                n_contrib[pix_id] = last_contributor;
                
                for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                    out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
                }
            }
        );
    });
    
    event.wait();
}

}
