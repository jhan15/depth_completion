import glob
import numpy as np
import os

from torch.utils.data import Dataset

from packnet_sfm.datasets.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from packnet_sfm.utils.image import load_image
from packnet_sfm.geometry.pose_utils import invert_pose_numpy

########################################################################################################################

IMAGE_FOLDER = 'camera_9/data'
LIDAR_FOLDER = 'proj_lidar/camera_9/data'
K_FILE = 'intrinsic.txt'

########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type].astype(np.float32)
    return np.expand_dims(depth, axis=2)

########################################################################################################################
#### DATASET
########################################################################################################################

class ScaniaDataset(Dataset):
    """
    Scania dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used
    batch_size : int
        Size of a batch to process
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context
    forward_context : int
        Number of forward frames to consider as context
    strides : tuple
        List of context strides
    """
    def __init__(self, root_dir, file_list, batch_size, excluded_folder=None, train=True,
                 data_transform=None, depth_type=None, input_depth_type=None, with_pose=False,
                 with_stereo=False, back_context=0, forward_context=0, strides=(1,)):
        # Assertions
        backward_context = back_context
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        self.split = file_list.split('/')[-1].split('.')[0]

        self.batch_size = batch_size
        self.excluded_folder = excluded_folder
        self.train = train
        self.root_dir = root_dir
        self.data_transform = data_transform

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.with_pose = with_pose
        self.with_stereo = with_stereo

        self.input_depth_type = input_depth_type
        self.with_input_depth = input_depth_type is not '' and input_depth_type is not None

        self._cache = {}
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        # Assume identical camera intrinsic
        self.intrinsic = np.loadtxt(os.path.join(self.root_dir, K_FILE)).astype(np.float32)

        with open(file_list, "r") as f:
            data = f.readlines()

        self.paths = []
        # Get file list from data
        for i, fname in enumerate(data):
            if self.excluded_folder and self.excluded_folder in fname:
                continue
            path = os.path.join(self.root_dir, fname.split()[0])
            add_flag = True
            if add_flag and self.with_input_depth:
                # Check if input depth file exists
                depth = self._get_depth_file(path)
                add_flag = depth is not None and os.path.exists(depth)
            if add_flag and self.with_depth:
                # Check if depth file exists
                depth = self._get_depth_file(path)
                add_flag = depth is not None and os.path.exists(depth)
            if add_flag:
                self.paths.append(path)

        # If using context, filter file list
        if self.with_context:
            paths_with_context = []
            for stride in strides:
                for idx, file in enumerate(self.paths):
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, backward_context, forward_context, stride)
                    if backward_context_idxs is not None and forward_context_idxs is not None:
                        paths_with_context.append(self.paths[idx])
                        self.forward_context_paths.append(forward_context_idxs)
                        self.backward_context_paths.append(backward_context_idxs[::-1])
            self.paths = paths_with_context

            # A temporary solution for an unresolved Pytorch 'merge_sort' issue when training with
            # SPARSE LiDAR gt and the training samples can not be evenly divided by the batch size
            remainder = len(self.paths) % self.batch_size
            if self.train and self.with_input_depth and remainder != 0:
                self.paths += self.paths[remainder - self.batch_size:]
                self.forward_context_paths += self.forward_context_paths[remainder - self.batch_size:]
                self.backward_context_paths += self.backward_context_paths[remainder - self.batch_size:]

########################################################################################################################

    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if depth_file.endswith('.npz'):
            depth = read_npz_depth(depth_file, self.depth_type)
            return depth
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    @staticmethod
    def _get_depth_file(image_file):
        """Get the corresponding depth file from an image file."""
        depth_file = image_file.replace(IMAGE_FOLDER, LIDAR_FOLDER)
        depth_file = depth_file.replace('png', 'npz')

        return depth_file
    
    @staticmethod
    def _sampling_depth(depth, percent=0.8):
        depth = depth.squeeze()
        rows, cols = np.where(depth>0)
        indices = np.arange(len(rows))
        np.random.shuffle(indices)

        for id in indices[:int((1-percent)*len(rows))]:
            depth[rows[id], cols[id]] = 0

        return np.expand_dims(depth, axis=2)

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths, None

########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {
            'idx': idx,
            'filename': '%s_%05d' % (self.split, idx),
            'rgb': load_image(self.paths[idx]),
        }

        # Add intrinsics
        sample.update({
            'intrinsics': self.intrinsic,
        })

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': self._read_depth(self._get_depth_file(self.paths[idx])),
            })

        # Add input depth information if requested
        if self.with_input_depth:
            sample.update({
                'input_depth': self._read_depth(self._get_depth_file(self.paths[idx])),
            })

        # Add context information if requested
        if self.with_context:
            # Add context images
            all_context_idxs = self.backward_context_paths[idx] + \
                               self.forward_context_paths[idx]
            image_context_paths, _ = \
                self._get_context_files(self.paths[idx], all_context_idxs)
            image_context = [load_image(f) for f in image_context_paths]
            sample.update({
                'rgb_context': image_context
            })

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        # Return sample
        return sample

########################################################################################################################
