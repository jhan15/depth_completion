import torch
import torch.nn as nn
from functools import partial
import MinkowskiEngine as ME

from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.layers.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth


class DepthResNetSAN(nn.Module):
    """
    DepthResNet-SAN network for thesis project @ Scania

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    grad_image_encoder : bool
        Requires grad for image encoder or not
    grad_image_decoder : bool
        Requires grad for image decoder or not
    grad_depth_eccoder : bool
        Requires grad for depth encoder or not
    scale_output : bool
        True if scaling the network output to [0.1, 100] units
    adjust_depth : bool
        True if adjust the lidar data by a transformation matrix
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, grad_image_encoder=True, grad_image_decoder=True,
                 grad_depth_encoder=True, scale_output=True, adjust_depth=False, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)
        
        # Image enconder and decoder
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        # Depth encoder
        self.mconvs = MinkowskiEncoder(self.encoder.num_ch_enc, with_uncertainty=False)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)
        
        # Requires grads or not
        for param in self.encoder.parameters():
            param.requires_grad = grad_image_encoder
        for param in self.decoder.parameters():
            param.requires_grad = grad_image_decoder
        for param in self.mconvs.parameters():
            param.requires_grad = grad_depth_encoder
        
        self.weight = nn.parameter.Parameter(torch.ones(5), requires_grad=grad_depth_encoder)
        self.bias = nn.parameter.Parameter(torch.zeros(5), requires_grad=grad_depth_encoder)

        self.scale_output = scale_output

        # Learnable transformation matrix: translation + axisangle
        self.pose = None
        if adjust_depth:
            self.pose = nn.parameter.Parameter(torch.zeros(6), requires_grad=True)

        # No weight initialization for SAN in PackNetSAN
        # TODO: Compare w/ and w/o weight initialization for SAN
        # self.init_weights()
    
    def init_weights(self):
        """Initializes SAN network weights."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def run_network(self, rgb, input_depth=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(rgb)

        if input_depth is not None:
            self.mconvs.prep(input_depth)

            x[0] = x[0] * self.weight[0].view(1, 1, 1, 1) + self.mconvs(x[0]) + self.bias[0].view(1, 1, 1, 1)
            x[1] = x[1] * self.weight[1].view(1, 1, 1, 1) + self.mconvs(x[1]) + self.bias[1].view(1, 1, 1, 1)
            x[2] = x[2] * self.weight[2].view(1, 1, 1, 1) + self.mconvs(x[2]) + self.bias[2].view(1, 1, 1, 1)
            x[3] = x[3] * self.weight[3].view(1, 1, 1, 1) + self.mconvs(x[3]) + self.bias[3].view(1, 1, 1, 1)
            x[4] = x[4] * self.weight[4].view(1, 1, 1, 1) + self.mconvs(x[4]) + self.bias[4].view(1, 1, 1, 1)
        
        x_dec = self.decoder(x)
        disps = [x_dec[('disp', i)] for i in range(4)]

        if self.training:
            inv_depths = [self.scale_inv_depth(d)[0] for d in disps] if self.scale_output else disps
        else:
            inv_depths = [self.scale_inv_depth(disps[0])[0]] if self.scale_output else [disps[0]]

        return inv_depths, x

    def forward(self, rgb, input_depth=None, **kwargs):

        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {'inv_depths': inv_depths}

        output = {}

        inv_depths_rgb, features_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb

        if input_depth is None:
            return output

        inv_depths_rgbd, features_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd

        loss = sum([((srgbd - srgb) ** 2).mean()
                    for srgbd, srgb in zip(features_rgbd, features_rgb)]) / len(features_rgbd)
        output['depth_loss'] = loss

        return output
