# Copyright 2020 Toyota Research Institute.  All rights reserved.

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.geometry.pose import Pose


class SelfSupModel(SfmModel):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    self_sup_signal : str
        Self-supervision signal, options are ['M', 'S', 'MS']
        M - monocular supervision, S - stereo supervision, MS - both
    kwargs : dict
        Extra parameters
    """
    def __init__(self, self_sup_signal='M', eval_self_sup_loss=False, **kwargs):
        # Initializes SfmModel
        super().__init__(**kwargs)
        self.self_sup_signal = self_sup_signal
        self.eval_self_sup_loss = eval_self_sup_loss
        # Pose network is only required if there is monocular self-supervision
        if 'M' not in self.self_sup_signal:
            self._network_requirements.remove('pose_net')
        # Initializes the photometric loss
        self._photometric_loss = MultiViewPhotometricLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs
        }

    def self_supervised_loss(self, image, ref_images, inv_depths, poses, intrinsics,
                             ref_intrinsics, eval_mode=False, return_logs=False, progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        ref_intrinsics : list of torch.Tensor [B,3,3]
            List of reference camera intrinsics
        eval_mode : bool
            True if evaluation mode
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._photometric_loss(
            image, ref_images, inv_depths, intrinsics, ref_intrinsics, poses,
            eval_mode=eval_mode, return_logs=return_logs, progress=progress)

    def forward(self, batch, rgbd=False, eval_loss=False, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        rgbd : bool
            True if using inverse depths predicted by rgb+d
        eval_loss : bool
            True if loss is calculated when not training
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            if eval_loss or self.eval_self_sup_loss:
                # Calculate photometric loss, use Stereo context only
                rgb_context = [batch['stereo_rgb_context']]
                pose_context = [Pose(batch['stereo_pose_context'])]
                ref_intrinsics = [batch['stereo_intrinsics']]
                self_sup_output = self.self_supervised_loss(
                    batch['rgb'], rgb_context, output['inv_depths'], pose_context, batch['intrinsics'],
                    ref_intrinsics, eval_mode=True, return_logs=return_logs, progress=progress)
                if 'rgb_warped' in self.logs:
                    batch['rgb_warped'] = self.logs['rgb_warped'][0][0]
                if 'ssim' in self.logs:
                    batch['ssim'] = self.logs['ssim'][0][0]
                output['photometric_loss'] = self_sup_output['loss'].detach()
            return output
        else:
            # Otherwise, calculate self-supervised loss
            inv_depths = output['inv_depths_rgbd'] if rgbd else output['inv_depths']
            rgb_context, pose_context, ref_intrinsics = [], [], []
            if 'S' in self.self_sup_signal:
                rgb_context += [batch['stereo_rgb_context_original']]
                pose_context += [Pose(batch['stereo_pose_context'])]
                ref_intrinsics += [batch['stereo_intrinsics']]
            if 'M' in self.self_sup_signal:
                rgb_context += batch['rgb_context_original']
                pose_context += output['poses']
                ref_intrinsics += [batch['intrinsics']] * 2
            # Calculate loss
            self_sup_output = self.self_supervised_loss(
                batch['rgb_original'], rgb_context, inv_depths, pose_context, batch['intrinsics'],
                ref_intrinsics, return_logs=return_logs, progress=progress)
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
