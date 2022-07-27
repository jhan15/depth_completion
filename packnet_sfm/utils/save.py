# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import os
import cv2

from packnet_sfm.utils.image import write_image
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth, scale_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix
from packnet_sfm.utils.types import is_tensor


def save_depth(batch, output, args, dataset, save, scale_fn):
    """
    Save depth predictions in various ways

    Parameters
    ----------
    batch : dict
        Batch from dataloader
    output : dict
        Output from model
    args : tuple
        Step arguments
    dataset : CfgNode
        Dataset configuration
    save : CfgNode
        Save configuration
    """
    # If there is no save folder, don't save
    if save.folder is '':
        return

    # If we want to save
    if save.depth.rgb or save.depth.viz or save.depth.npz or save.depth.png:
        # Retrieve useful tensors
        rgb = batch['rgb']
        rgb_warped = batch['rgb_warped'] if 'rgb_warped' in batch else None
        ssim = batch['ssim'] if 'ssim' in batch else None
        pred_inv_depth = output['inv_depth']
        depth = batch['input_depth'] if 'input_depth' in batch else None
        depth_gt = batch['depth'] if 'depth' in batch else None

        # Prepare path strings
        filename = batch['filename']
        dataset_idx = 0 if len(args) == 1 else args[1]
        save_path = os.path.join(save.folder, 'depth',
                                 prepare_dataset_prefix(dataset, dataset_idx),
                                 os.path.basename(save.pretrained).split('.')[0])
        # Create folder
        os.makedirs(save_path, exist_ok=True)

        # For each image in the batch
        length = rgb.shape[0]
        for i in range(length):
            # Save numpy depth maps
            if save.depth.npz:
                rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy()
                viz_i = viz_inv_depth(pred_inv_depth[i], normalizer=0.17)
                if is_tensor(depth):
                    invd_i = 1. / depth[i]
                    invd_i[depth[i] <= 0] = 0
                    viz_invd_i = viz_inv_depth(invd_i, filter_zeros=True, zero_to_nan=True, colormap='hsv')
                else:
                    viz_invd_i = None
                write_depth('{}/{}_depth.npz'.format(save_path, filename[i]),
                            depth=inv2depth(pred_inv_depth[i]),
                            intrinsics=batch['intrinsics'][i] if 'intrinsics' in batch else None,
                            rgb=rgb_i, viz=viz_i, depth_input=viz_invd_i)
            # Save png depth maps
            if save.depth.png:
                write_depth('{}/{}_depth.png'.format(save_path, filename[i]),
                            depth=inv2depth(pred_inv_depth[i]))
            # Save rgb images
            if save.depth.rgb:
                rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                write_image('{}/{}_rgb.png'.format(save_path, filename[i]), rgb_i)
            # Save inverse depth visualizations
            if save.depth.viz:
                viz_i = viz_inv_depth(pred_inv_depth[i], colormap='hsv', normalizer=0.17) * 255
                write_image('{}/{}_viz.png'.format(save_path, filename[i]), viz_i)

            # Save depth gt
            if is_tensor(depth_gt) and save.depth.dgt:
                inv_i = 1. / depth_gt[i]
                inv_i[depth_gt[i] <= 0] = 0
                viz_i = viz_inv_depth(inv_i, filter_zeros=True, zero_to_nan=True, colormap='hsv', normalizer=0.17) * 255
                # viz_i[np.where((viz_i==[0,0,0]).all(axis=2))] = [255,255,255]
                write_image('{}/{}_depth_gt.png'.format(save_path, filename[i]), viz_i)

            # Save depth error map
            if is_tensor(depth_gt) and save.depth.dem:
                inv_depth_up = scale_depth(pred_inv_depth, depth_gt, scale_fn)
                abs_rel_error = get_error_map(inv_depth_up[i], depth_gt[i])
                viz_i = viz_inv_depth(abs_rel_error, normalizer=0.1, colormap='Reds') * 255
                write_image('{}/{}_error_map.png'.format(save_path, filename[i]), viz_i)
            
            # Save photometric error map
            if is_tensor(rgb_warped) and save.depth.pem:
                rgb_i = rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                rgb_warped_i = rgb_warped[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                ssim_i = 255 - ssim[i].permute(1, 2, 0).detach().cpu().numpy() * 255
                # ssim_i = cv2.cvtColor(ssim_i, cv2.COLOR_RGB2GRAY)
                # cv2.imwrite('{}/{}_ssim.png'.format(save_path, filename[i]), ssim_i)
                abs_error_i = 255 - cv2.absdiff(rgb_i, rgb_warped_i)
                write_image('{}/{}_photometric_error.png'.format(save_path, filename[i]), abs_error_i)
                write_image('{}/{}_ssim.png'.format(save_path, filename[i]), ssim_i)


def get_error_map(inv_depth, depth_gt, filter_zeros=True):
    """
    Compute absolute relative error map.

    Parameters
    ----------
    inv_depth : torch.tensor [1,1,H,W]
        Predicted inverse depth map
    depth_gt : np.array [H,W]
        Ground truth depth map
    filter_zeros : bool
        If True, convert zero values to nan or inf

    Returns
    -------
    abs_rel_error : np.array [H,W]
        Absolute relative error map
    """
    # Convert to depth
    depth = inv2depth(inv_depth).detach().squeeze().cpu().numpy()
    depth_gt = depth_gt.detach().squeeze().cpu().numpy()
    # Mask out invalid pixels
    depth_gt[depth_gt <= 0] = 0.
    depth[depth_gt == 0] = 0.
    # Calculate absolute relative errors
    if not filter_zeros:
        abs_rel_error = np.divide(np.absolute(depth - depth_gt), depth_gt,
                                  out=np.zeros_like(depth_gt), where=depth_gt!=0)
    else:
        abs_rel_error = np.absolute(depth - depth_gt) / depth_gt

    return abs_rel_error
