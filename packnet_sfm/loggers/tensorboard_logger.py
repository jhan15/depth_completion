
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from packnet_sfm.utils.depth import viz_inv_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix
from packnet_sfm.utils.types import is_dict, is_tensor

class TensorBoardLogger:
    def __init__(self, log_dir = None):
        super().__init__()
        self._writer = SummaryWriter(log_dir, comment="hej")
        self._metrics = OrderedDict()
        self._images = []

    def log_metrics(self, metrics):
        """Logs training metrics. Buffers until global_step is included."""
        # Ugly code here because we want to fit into framework
        self._metrics.update(metrics)
        if 'global_step' in metrics:
            # Include the metrics of step 0 if validate_first flag set to True
            if 'avg_train-loss' not in self._metrics:
                global_step = 0
            else:
                global_step = metrics['global_step']
            # Untangle global prefix
            known_key = '-abs_rel' # picked at random
            prefix_list = []
            for key, value in self._metrics.items():
                if key.endswith(known_key):
                    prefix = key[:-len(known_key)]
                    prefix_list.append(prefix)
            # Metrics that we want to save
            metrics_keys = ('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
            # ...and their variations
            metrics_modes = ('', '_pp', '_gt', '_pp_gt')
            # Losses to log, photometric loss only for now
            loss_keys = ['photometric_loss']
            for prefix in prefix_list:
                for key in loss_keys:
                    full_key = prefix + '-' + key
                    if full_key in self._metrics:
                        tag_scalar_dict = dict()
                        tag_scalar_dict[key] = self._metrics[full_key]
                        main_tag = prefix + '/_' + key
                        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)
                for key in metrics_keys:
                    tag_scalar_dict = dict()
                    for mode in metrics_modes:
                        full_key = prefix + '-' + key + mode
                        if full_key in self._metrics:
                            tag_scalar_dict[key+mode] = self._metrics[full_key]
                    main_tag = prefix + '/' + key
                    self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)

            for key, value in self._metrics.items():
                if key.startswith('avg_train') or key.endswith('learning_rate'):
                    self._writer.add_scalar('Training/' + key, value, global_step)
            
            # Log feature fusion weight and bias
            if 'w_0' in self._metrics: # picked at random
                weight, bias = dict(), dict()
                weight_keys = ('w_0', 'w_1', 'w_2', 'w_3', 'w_4')
                bias_keys = ('b_0', 'b_1', 'b_2', 'b_3', 'b_4')
                for key in weight_keys:
                    weight[key] = self._metrics[key]
                for key in bias_keys:
                    bias[key] = self._metrics[key]
                self._writer.add_scalars('Weight/depth_net.weight', weight, global_step)
                self._writer.add_scalars('Weight/depth_net.bias', bias, global_step)

            for i in self._images:
                tag = i['prefix'] + '/' + i['key']
                self._writer.add_image(tag, i['image'], global_step, dataformats='HWC')

            self._images = []
            self._metrics.clear()

    # Log depth images
    def log_depth(self, *args, **kwargs):
        """Helper function used to log images relevant for depth estimation"""
        def log(prefix_idx, batch, output):
            self._images.append(log_rgb('rgb', prefix_idx, batch))
            if 'inv_depth' in output:
                self._images.append(log_inv_depth('inv_depth', prefix_idx, output))
            if 'depth' in batch:
                self._images.append(log_depth('depth', prefix_idx, batch))
            if 'rgb_warped' in output:
                self._images.append(log_warped_rgb('rgb_warped', prefix_idx, output))
            if 'rgb_ref' in output:
                self._images.append(log_ref_rgb('rgb_ref', prefix_idx, output))
        self.log_images(log, *args, **kwargs)

    def log_images(self, func, mode, batch, output,
                   args, dataset, world_size, config):
        """
        Adds images to buffer for later logging.

        Parameters
        ----------
        func : Function
            Function used to process the image before logging
        mode : str {"train", "val"}
            Training stage where the images come from (serve as prefix for logging)
        batch : dict
            Data batch
        output : dict
            Model output
        args : tuple
            Step arguments
        dataset : CfgNode
            Dataset configuration
        world_size : int
            Number of GPUs, used to get logging samples at consistent intervals
        config : CfgNode
            Model configuration
        """
        dataset_idx = 0 if len(args) == 1 else args[1]
        prefix = prepare_dataset_prefix(config, dataset_idx)
        interval = len(dataset[dataset_idx]) // world_size // config.num_logs
        if mode == 'train':
            for i, idx in enumerate(batch['idx'].tolist()):
                if (idx % interval) == 0:
                    prefix_idx = '{}-{}-{}'.format(mode, prefix, batch['idx'][i].item())
                    func(prefix_idx, batch, output)
        else:
            if args[0] % interval == 0:
                prefix_idx = '{}-{}-{}'.format(mode, prefix, batch['idx'][0].item())
                func(prefix_idx, batch, output)

def log_rgb(key, prefix, batch, i=0):
    """
    Converts an RGB image from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the image
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the image

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    rgb = batch[key] if is_dict(batch) else batch
    return prep_image(prefix, key,
                      rgb[i])


def log_warped_rgb(key, prefix, batch, i=0):
    """
    Converts warped RGB images from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the warped images
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the image

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    warped_rgb = batch[key] if is_dict(batch) else batch
    warped_rgb = [warped_rgb[j][-1][i] for j in range(len(warped_rgb))]
    warped_rgb = [w.detach().permute(1, 2, 0).cpu().numpy() for w in warped_rgb]
    warped_rgb = np.concatenate(warped_rgb, 0)
    return prep_image(prefix, key, warped_rgb)


def log_ref_rgb(key, prefix, batch, i=0):
    """
    Converts reference RGB images from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the reference images
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the image

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    ref_rgb = batch[key] if is_dict(batch) else batch
    ref_rgb = [ref_rgb[j][i] for j in range(len(ref_rgb))]
    ref_rgb = [r.detach().permute(1, 2, 0).cpu().numpy() for r in ref_rgb]
    ref_rgb = np.concatenate(ref_rgb, 0)
    return prep_image(prefix, key, ref_rgb)


def log_depth(key, prefix, batch, i=0):
    """
    Converts a depth map from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the depth map
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the depth map

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    depth = batch[key] if is_dict(batch) else batch
    image = batch['rgb'][i].permute(1, 2, 0).detach().cpu().numpy()
    inv_depth = 1. / depth[i]
    inv_depth[depth[i] == 0] = 0
    viz = viz_inv_depth(inv_depth, normalizer=0.17, filter_zeros=True, zero_to_nan=True)
    viz[np.where((viz==[0,0,0]).all(axis=2))] = image[np.where((viz==[0,0,0]).all(axis=2))]
    return prep_image(prefix, key, viz)


def log_inv_depth(key, prefix, batch, i=0):
    """
    Converts an inverse depth map from a batch for logging

    Parameters
    ----------
    key : str
        Key from data containing the inverse depth map
    prefix : str
        Prefix added to the key for logging
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the inverse depth map

    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    inv_depth = batch[key] if is_dict(batch) else batch
    return prep_image(prefix, key,
                      viz_inv_depth(inv_depth[i], normalizer=0.17))


def prep_image(prefix, key, image):
    """
    Prepare image for wandb logging

    Parameters
    ----------
    prefix : str
        Prefix added to the key for logging
    key : str
        Key from data containing the inverse depth map
    image : torch.Tensor [3,H,W]
        Image to be logged

    Returns
    -------
    output : dict
        Dictionary with key and value for logging
    """
    if is_tensor(image):
        image = image.detach().permute(1, 2, 0).cpu().numpy()
    return {'prefix' : prefix, 'key' : key, 'image' : image}

