cmake_minimum_required(VERSION 3.20)

project(IntelDiffGaussianRasterization LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Intel oneAPI DPC++ compiler
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx" AND NOT CMAKE_CXX_COMPILER MATCHES "dpcpp")
    message(WARNING "Intel DPC++ compiler not detected. Searching for icpx...")
    find_program(ICPX_COMPILER icpx)
    if(ICPX_COMPILER)
        set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
        message(STATUS "Found Intel DPC++ compiler: ${ICPX_COMPILER}")
    else()
        message(FATAL_ERROR "Intel DPC++ compiler (icpx) not found. Please source oneAPI environment.")
    endif()
endif()

# SYCL-specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# Intel Arc-specific optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Optional: Enable specific SYCL device
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=intel_gpu_pvc")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sycl_rasterizer)

# Add the SYCL rasterizer library
add_library(SyclRasterizer
    sycl_rasterizer/forward.dp.cpp
    sycl_rasterizer/rasterizer_impl.dp.cpp
)

# Link SYCL library
target_link_libraries(SyclRasterizer 
    sycl
)

# Set target properties
set_target_properties(SyclRasterizer PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)

# Optional: Create output directory for intermediate files
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# Optional: Generate SPIR-V for inspection
add_custom_command(
    TARGET SyclRasterizer
    POST_BUILD
    COMMAND ${CMAKE_CXX_COMPILER} -fsycl -fsycl-device-only -c 
            ${CMAKE_SOURCE_DIR}/sycl_rasterizer/forward.dp.cpp 
            -o ${CMAKE_SOURCE_DIR}/out/forward.spv || true
    COMMENT "Generating SPIR-V for forward.dp.cpp"
    VERBATIM
)

# Installation
install(TARGETS SyclRasterizer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY sycl_rasterizer/
    DESTINATION include/sycl_rasterizer
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "=================================================")
message(STATUS "Intel XPU Gaussian Rasterization Configuration")
message(STATUS "=================================================")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "=================================================")
