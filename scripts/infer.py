# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imwrite
from matplotlib.cm import get_cmap

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, resize_depth, to_tensor, \
    crop_image, crop_depth, resize_depth_preserve
from packnet_sfm.datasets.kitti_dataset import read_npz_depth
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import load_depth, write_depth, inv2depth, depth2inv, \
    viz_inv_depth, scale_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.utils.misc import parse_crop_borders


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    # Inputs
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint (.ckpt)')
    parser.add_argument('--image', type=str, required=True, help='Input image file or folder')
    parser.add_argument('--depth', type=str, help='Input depth file or folder')
    parser.add_argument('--depth_gt', type=str, help='Depth ground truth file or folder')
    parser.add_argument('--intrinsics', type=str, help='Camera intrinsics file')
    # Inference output
    parser.add_argument('--output', type=str, required=True, help='Output file or folder')
    # Customized outputs
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    parser.add_argument('--stack_depth_input', action='store_true', help='Stack depth input to output file')
    # Other args
    parser.add_argument('--half', action='store_true', help='Use half precision (fp16)')
    args = parser.parse_args()

    return args


@torch.no_grad()
# fn, model_wrapper, image_shape, image_crop, input_depth_type, args
def infer_and_save_depth(input_file, model_wrapper, image_shape, image_crop,
                         input_depth_type, args):
    image_file, depth_file, depth_gt_file = input_file
    
    # Check if the image and depth files match
    if depth_file is not None:
        image_prefix = image_file.split('/')[-1].split('.')[0]
        depth_prefix = depth_file.split('/')[-1].split('.')[0]
        if image_prefix != depth_prefix:
            return
    
    if not is_image(args.output):
        # If not an image, assume it's a folder and append the image input name
        os.makedirs(args.output, exist_ok=True)
        output = os.path.join(args.output, os.path.basename(image_file))
    else:
        output = args.output

    dtype = torch.float16 if args.half else None

    # Load inputs
    image = load_image(image_file)
    depth = read_npz_depth(depth_file, input_depth_type) if depth_file else None
    depth_gt = load_depth(depth_gt_file) if depth_gt_file else None
    intrinsics = np.loadtxt(args.intrinsics) if args.intrinsics else None

    # Apply augmentation to inputs
    if image_shape:
        scale_fn = 'resize'
        image_in = resize_image(image, image_shape)
        depth_in = resize_depth_preserve(depth, image_shape) if depth_file else None
        
    if image_crop:
        scale_fn = 'top-center'
        borders = parse_crop_borders(image_crop, np.array(image).shape[:2])
        image_in = crop_image(image, borders)
        depth_in = crop_depth(depth, borders) if depth_file else None
    
    image_in = to_tensor(image_in).unsqueeze(0)
    depth_in = to_tensor(depth_in).unsqueeze(0) if depth_file else None
    depth_gt_tensor = to_tensor(depth_gt).unsqueeze(0) if depth_gt_file else None
    depth_tensor = to_tensor(depth).unsqueeze(0) if depth_file else None
    if torch.cuda.is_available():
        image_in = image_in.to('cuda:{}'.format(rank()), dtype=dtype)
        depth_in = depth_in.to('cuda:{}'.format(rank()), dtype=dtype) if depth_file else None

    # Infer
    pred_inv_depth = []
    pred_inv_depth.append(model_wrapper.depth(image_in)['inv_depths'][0])
    if depth_file:
        pred_inv_depth.append(model_wrapper.depth(image_in, depth_in)['inv_depths'][0])

    # Save a png depth map
    if args.save == 'png':
        filename = '{}.{}'.format(os.path.splitext(output)[0], args.save)
        print('Saving {} to {}'.format(
            pcolor(image_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth[-1]))
    
    # Save a npz depth map (along with other attributes for 3d reprojection)
    elif args.save == 'npz':
        assert args.depth_gt is not None, 'Need depth ground truth for scaling depth'
        # Convert data to target formats
        inv_depth = scale_depth(pred_inv_depth[-1], depth_gt_tensor, scale_fn)
        pred_depth = inv2depth(inv_depth).detach().squeeze().cpu().numpy()
        viz = viz_inv_depth(inv_depth[0], normalizer=0.17)
        viz_depth_input = viz_inv_depth(depth2inv(depth_tensor)[0], filter_zeros=True,
            zero_to_nan=True, colormap='hsv') if depth_file else None

        filename = '{}.{}'.format(os.path.splitext(output)[0], args.save)
        print('Saving {} to {}'.format(
            pcolor(image_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=pred_depth, intrinsics=intrinsics,
                    rgb=np.array(image)/255, viz=viz, depth_input=viz_depth_input)
    
    # Save an inference stacked by input and prediction
    else:
        image = image_in[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare input depth if provided
        if depth_file and args.stack_depth_input:
            viz_depth_input = viz_inv_depth(depth2inv(depth_in)[0], filter_zeros=True,
                                            zero_to_nan=True, colormap='hsv') * 255
            viz_depth_input[np.where((viz_depth_input==[0,0,0]).all(axis=2))] = [255,255,255]
            image = np.concatenate([image, viz_depth_input], 0)
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[-1][0], colormap='hsv') * 255
        image = np.concatenate([image, viz_pred_inv_depth], 0)
        pred_depth = inv2depth(pred_inv_depth[-1]).detach().squeeze().cpu().numpy()
        # Save visualization
        print('Saving {} to {} (depth median: {})'.format(
            pcolor(image_file, 'cyan', attrs=['bold']),
            pcolor(output, 'magenta', attrs=['bold']),
            pcolor(np.median(pred_depth), 'magenta', attrs=['bold'])))
        imwrite(output, image[:, :, ::-1])


def main(args):

    hvd_init()
    config, state_dict = parse_test_file(args.checkpoint)

    # A temp fix for using pre-trained packnetsan checkpoint, there is a network name mismatch
    if config.model.depth_net.name == 'PackNetSlimEnc01':
        config.model.depth_net.name = 'PackNetSAN01'

    # The config file in the checkpoint should either has image_shape or crop_train_borders
    image_shape = config.datasets.augmentation.image_shape
    image_crop = config.datasets.augmentation.crop_train_borders
    input_depth_type = config.datasets.train.input_depth_type[0]

    assert (len(image_shape) == 0 and len(image_crop) > 0) or \
        (len(image_shape) > 0 and len(image_crop) == 0), \
        'Both image_shape and crop_train_board are not defined in the checkpoint config file'

    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    dtype = torch.float16 if args.half else None
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)
    model_wrapper.eval()

    if os.path.isdir(args.image):
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.image, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} image files'.format(len(files)))
    else:
        files = [args.image]
    
    if args.depth:
        if os.path.isdir(args.depth):
            depth_files = []
            depth_files.extend(glob((os.path.join(args.depth, '*.{}'.format('npz')))))
            depth_files.sort()
            print0('Found {} depth files'.format(len(depth_files)))
        else:
            depth_files = [args.depth]
    else:
        depth_files = [None] * len(files)
    
    if args.depth_gt:
        if os.path.isdir(args.depth_gt):
            depth_gt_files = []
            depth_gt_files.extend(glob((os.path.join(args.depth_gt, '*.{}'.format('png')))))
            depth_gt_files.sort()
            print0('Found {} depth gt files'.format(len(depth_gt_files)))
        else:
            depth_gt_files = [args.depth_gt]
    else:
        depth_gt_files = [None] * len(files)
    
    files = list(zip(files, depth_files, depth_gt_files))

    # Process each file
    for fn in files[rank()::world_size()]:
        infer_and_save_depth(fn, model_wrapper, image_shape, image_crop, input_depth_type, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
