#ifndef SYCL_RASTERIZER_CONFIG_H_INCLUDED
#define SYCL_RASTERIZER_CONFIG_H_INCLUDED

// Rendering configuration
#define NUM_CHANNELS 3

// Tile config
// 16 threads per XVE
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y) //256 threads per workgroup

// Sub group config
#define SUBGROUP_SIZE 16
#define NUM_SUBGROUPS (BLOCK_SIZE / SUBGROUP_SIZE) // 16 Subgroups per workgroup

// Max no of gaussians per tile(for pre-allocation)
#define MAX_GAUSSIANS_PER_TILE 4096

// Sorting config
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS) // 256

#endif
