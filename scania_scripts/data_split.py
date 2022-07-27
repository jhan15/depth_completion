import os
import argparse
from glob import glob
import numpy as np

np.random.seed(42)
image_key = 'camera_9/data/*.png'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--split_ratio', type=float, nargs=3, default=(0.7,0.2,0.1))
    args = parser.parse_args()

    return args

def main(args):
    folders = next(os.walk(args.path))[1]
    image_paths = []

    for folder in folders:
        image_folder = os.path.join(args.path, folder, image_key)
        image_paths += glob(image_folder)

    image_paths = [image_path.replace(args.path, '') for image_path in image_paths]
    image_paths.sort()
    
    # exclude a small sequence for demo
    image_paths = image_paths[:-329]

    np.random.shuffle(image_paths)

    x, y, z = args.split_ratio
    n = len(image_paths)

    train_dataset = image_paths[:int(x*n)]
    val_dataset   = image_paths[int(x*n):int((x+y)*n)]
    test_dataset  = image_paths[int((x+y)*n):]

    print('data split')
    print('-- train:\t', len(train_dataset))
    print('-- valid:\t', len(val_dataset))
    print('-- test:\t', len(test_dataset))

    train_save = os.path.join(args.path, 'train.txt')
    val_save = os.path.join(args.path, 'val.txt')
    test_save = os.path.join(args.path, 'test.txt')
    np.savetxt(train_save, train_dataset, delimiter=' ', fmt='%s')
    np.savetxt(val_save, val_dataset, delimiter=' ', fmt='%s')
    np.savetxt(test_save, test_dataset, delimiter=' ', fmt='%s')

if __name__ == '__main__':
    args = parse_args()
    main(args)
