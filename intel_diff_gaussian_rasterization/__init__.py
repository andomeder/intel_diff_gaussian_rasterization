#
# Intel XPU Differential Gaussian Rasterization
# Based on the original work by INRIA GRAPHDECO research group
#

from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    """Deep copy tuple to CPU"""
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item 
        for item in input_tuple
    ]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    """Rasterize 3D Gaussians using Intel XPU"""
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments for C++ interface
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke Intel XPU rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                    _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nError in forward pass. Saved snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
                _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp, means3D, scales, rotations, cov3Ds_precomp, 
            radii, sh, geomBuffer, binningBuffer, imgBuffer
        )
        
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, \
            geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D, 
            radii, 
            colors_precomp, 
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            cov3Ds_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_color, 
            sh, 
            raster_settings.sh_degree, 
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug
        )

        # Compute gradients by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
                    grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
                    _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nError in backward pass. Saved snapshot_bw.dump for debugging.")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
                grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
                _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster_settings
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    """Settings for Gaussian rasterization"""
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    """Gaussian Rasterizer module for Intel XPU"""
    
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """Mark visible points based on frustum culling"""
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )
        return visible

    def forward(
        self, 
        means3D, 
        means2D, 
        opacities, 
        shs=None, 
        colors_precomp=None, 
        scales=None, 
        rotations=None, 
        cov3D_precomp=None
    ):
        """Forward pass of Gaussian rasterization"""
        raster_settings = self.raster_settings

        # Validation
        if (shs is None and colors_precomp is None) or \
           (shs is not None and colors_precomp is not None):
            raise Exception('Provide exactly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # Handle empty tensors
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke rasterization
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )


# Export public API
__all__ = [
    'GaussianRasterizationSettings',
    'GaussianRasterizer',
    'rasterize_gaussians',
]
