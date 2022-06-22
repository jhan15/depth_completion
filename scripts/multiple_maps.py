import argparse
import numpy as np
import os
from glob import glob
from packnet_sfm.utils.depth import write_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Multiple depth maps into one file')
    parser.add_argument('--input', type=str, required=True, help='Depth maps folder')
    parser.add_argument('--output', type=str, required=True, help='Output npz file')
    args = parser.parse_args()

    return args


def main(args):
    depth_files = []
    depth_files.extend(glob((os.path.join(args.input, '*.{}'.format('npz')))))
    depth_files.sort()

    rgb = []
    viz = []
    depth = []
    depth_input = []
    intrinsics = None

    for depth_file in depth_files:
        data = np.load(depth_file, allow_pickle=True)
        if intrinsics is None:
            intrinsics = data['intrinsics']
        rgb.append(data['rgb'])
        viz.append(data['viz'])
        depth.append(data['depth'])
        depth_input.append(data['depth_input'])

    write_depth(args.output, depth=depth, intrinsics=intrinsics,
                rgb=rgb, viz=viz, depth_input=depth_input)


if __name__ == '__main__':
    args = parse_args()
    main(args)
