# Implementation for MGMA-Net.
# 
# Code partially referenced from:
# https://github.com/facebookresearch/VMZ
# https://github.com/kenshohara/3D-ResNets-PyTorch

import torch
import torch.nn as nn
import numpy as np

from lib.models.batchnorm_helper import get_norm
from lib.models.mgma_helper import MGMA
from lib.models.operators import SE, Swish
from .build import MODEL_REGISTRY

def build_conv(block, 
            in_filters, 
            out_filters, 
            kernels, 
            strides=(1, 1, 1), 
            pads=(0, 0, 0), 
            conv_idx=1, 
            block_type="3d",
            norm_module=nn.BatchNorm3d,
            freeze_bn=False):
    
    if block_type == "2.5d":
        i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
        i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
        middle_filters = int(i)
        
        # 1x3x3 layer
        conv_middle = "conv{}_middle".format(str(conv_idx))
        block[conv_middle] = nn.Conv3d(
            in_filters,
            middle_filters,
            kernel_size=(1, kernels[1], kernels[2]),
            stride=(1, strides[1], strides[2]),
            padding=(0, pads[1], pads[2]),
            bias=False)
        
        bn_middle = "bn{}_middle".format(str(conv_idx))
        block[bn_middle] = norm_module(num_features=middle_filters, track_running_stats=(not freeze_bn))
        
        relu_middle = "relu{}_middle".format(str(conv_idx))
        block[relu_middle] = nn.ReLU(inplace=True)
        
        # 3x1x1 layer
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            middle_filters,
            out_filters,
            kernel_size=(kernels[0], 1, 1),
            stride=(strides[0], 1, 1),
            padding=(pads[0], 0, 0),
            bias=False)
    elif block_type == '0.3d' or block_type == '0.3d+relu':
        conv_middle = "conv{}_middle".format(str(conv_idx))
        block[conv_middle] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False)
        
        bn_middle = "bn{}_middle".format(str(conv_idx))
        block[bn_middle] = norm_module(num_features=out_filters, track_running_stats=(not freeze_bn))
        
        if block_type == '0.3d+relu':
            relu_middle = "relu{}_middle".format(str(conv_idx))
            block[relu_middle] = nn.ReLU(inplace=True)
            
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            out_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            groups=out_filters,
            bias=False)
    elif block_type == "3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)
    elif block_type == '3d-sep':
        assert in_filters == out_filters
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            groups=in_filters,
            bias=False)
    elif block_type == "i3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)

def build_basic_block(block, 
                    input_filters, 
                    output_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    build_conv(block,
            input_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides=strides, 
            pads=(1, 1, 1),
            conv_idx=1,
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    block["bn1"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    block["relu"] = nn.ReLU(inplace=True)
    
    build_conv(block,
            output_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides = (1, 1, 1),
            pads=(1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    block["bn2"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))

def build_bottleneck(block, 
                    input_filters, 
                    output_filters, 
                    base_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        if block_type == 'i3d':
            strides = (1, 2, 2)
        else:
            strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    temp_conv_size = 3 if block_type == 'i3d' and use_temp_conv else 1
    
    build_conv(block,
            input_filters, 
            base_filters, 
            kernels=(temp_conv_size, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            pads=(temp_conv_size // 2, 0, 0) if block_type == 'i3d' else (0, 0, 0),
            conv_idx=1,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    block["bn1"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    block["relu1"] = nn.ReLU(inplace=True)
    
    build_conv(block,
            base_filters, 
            base_filters, 
            kernels=(1, 3, 3) if block_type == 'i3d' else (3, 3, 3), 
            strides=strides,
            pads=(0, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    block["bn2"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    block["relu2"] = nn.ReLU(inplace=True)
    
    build_conv(block,
            base_filters, 
            output_filters, 
            kernels=(1, 1, 1), 
            conv_idx=3,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    block["bn3"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))

class BasicBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 use_temp_conv=False,
                 se_ratio=0.0,
                 down_sampling=False, 
                 block_type='3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(BasicBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.down_sampling = down_sampling
        self.se_ratio = se_ratio
        self.block = nn.ModuleDict()
        
        build_basic_block(self.block, 
                          input_filters, 
                          output_filters, 
                          use_temp_conv, 
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.se_ratio > 0.0:
            self.se = SE(output_filters, se_ratio)
        
    def forward(self, x):
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)
            
            if k == "bn2" and self.se_ratio > 0.0:
                out = self.se(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu"](out)
        
        return out

class MGMABasicBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 mgma_config,
                 use_temp_conv=False,
                 se_ratio=0.0,
                 down_sampling=False, 
                 block_type='3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(MGMABasicBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.mgma_config = mgma_config
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_basic_block(self.block, 
                          input_filters, 
                          output_filters, 
                          use_temp_conv,
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if len(mgma_config) > 0:
            self.mgma = MGMA(input_filters, output_filters, mgma_config, freeze_bn=freeze_bn, 
                             temporal_downsample=down_sampling, spatial_downsample=down_sampling)
            self.mgma_use_shuffle = mgma_config['USE_SHUFFLE'] if 'USE_SHUFFLE' in mgma_config else True
    
    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return torch.cat((x[0], x[1]), 1) 
        
    def forward(self, x):
        if len(self.mgma_config) > 0:
            if self.mgma_use_shuffle:
                x = self.channel_shuffle(x)
            mgma_out = x
        
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu"](out)
        
        if len(self.mgma_config) > 0:
            mgma_out = self.mgma(mgma_out)
            mgma_out = out * mgma_out
            out = out + mgma_out

        return out

class Bottleneck(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 use_temp_conv=False,
                 se_ratio=0.0,
                 down_sampling=False, 
                 block_type='3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(Bottleneck, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.base_filters = base_filters
        self.down_sampling = down_sampling
        self.se_ratio = se_ratio
        self.block = nn.ModuleDict()
        
        build_bottleneck(self.block, 
                          input_filters, 
                          output_filters, 
                          base_filters, 
                          use_temp_conv, 
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.se_ratio > 0.0:
            self.se = SE(output_filters, se_ratio)
        
    def forward(self, x):
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)
            
            if k == "bn3" and self.se_ratio > 0.0:
                out = self.se(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu1"](out)
        
        return out

class MGMABottleneck(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 mgma_config,
                 use_temp_conv=False,
                 down_sampling=False, 
                 block_type='3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(MGMABottleneck, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.base_filters = base_filters
        self.mgma_config = mgma_config
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_bottleneck(self.block, 
                          input_filters, 
                          output_filters, 
                          base_filters, 
                          use_temp_conv, 
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if len(mgma_config) > 0:
            self.mgma = MGMA(input_filters, output_filters, mgma_config, freeze_bn=freeze_bn, 
                             temporal_downsample=down_sampling, spatial_downsample=down_sampling)
            self.mgma_use_shuffle = mgma_config['USE_SHUFFLE'] if 'USE_SHUFFLE' in mgma_config else True
    
    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return torch.cat((x[0], x[1]), 1) 
        
    def forward(self, x):
        if len(self.mgma_config) > 0:
            if self.mgma_use_shuffle:
                x = self.channel_shuffle(x)
            mgma_out = x
        
        residual = x
        out = x
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        out += residual
        out = self.block["relu1"](out)
        
        if len(self.mgma_config) > 0:
            mgma_out = self.mgma(mgma_out)
            mgma_out = out * mgma_out
            out = out + mgma_out

        return out
    
@MODEL_REGISTRY.register()
class MGMANet(nn.Module):

    def __init__(self, cfg):
        
        model_name = cfg.MODEL.ARCH
        model_depth = cfg.MODEL.DEPTH
        sample_duration = cfg.DATA.NUM_FRAMES
        freeze_bn = cfg.BN.FREEZE_PARAMS
        num_classes = cfg.MODEL.NUM_CLASSES
        self.model_name = model_name
        self.sample_duration = sample_duration
        self.norm_module = get_norm(cfg)
        self.se_ratio = cfg.MODEL.SE_RATIO
        self.cfg = cfg
        
        if cfg.MGMA.TYPE != "NONE":
            self.mgma_config = cfg.MGMA
        else:
            self.mgma_config = {}
        
        layers, block, block_type, filter_config, final_temporal_kernel, final_spatial_kernel = obtain_arc(model_name, model_depth, sample_duration)
        
        super(MGMANet, self).__init__()
        
        if model_name in ["r3d", "i3d"] or "_csn" in model_name:
            self.conv1 = nn.Conv3d(
                3,
                64,
                kernel_size=(5, 7, 7) if model_name == "i3d" else (3, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3) if model_name == "i3d" else (1, 3, 3),
                bias=False)
            self.bn1 = self.norm_module(num_features=64, track_running_stats=(not freeze_bn))
            self.relu = nn.ReLU(inplace=True)
            if model_name == "i3d" or "_csn" in model_name:
                self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        else:    
            self.conv1_middle = nn.Conv3d(
                3,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False)
            self.bn1_middle = self.norm_module(num_features=45, track_running_stats=(not freeze_bn))
            self.relu = nn.ReLU(inplace=True)
            
            self.conv1 = nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False)
            self.bn1 = self.norm_module(num_features=64, track_running_stats=(not freeze_bn))

        self.layer1 = self._make_layer(block_func=block, layer_id=1, 
                                       num_blocks=layers[0], 
                                       input_filters=64, 
                                       output_filters=filter_config[0][0], 
                                       base_filters=int(filter_config[0][1]), 
                                       block_type=block_type,
                                       freeze_bn=freeze_bn)
        
        if model_name == "i3d":
            self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        
        self.layer2 = self._make_layer(block_func=block, layer_id=2, 
                                       num_blocks=layers[1], 
                                       input_filters=filter_config[0][0], 
                                       output_filters=filter_config[1][0], 
                                       base_filters=int(filter_config[1][1]), 
                                       block_type=block_type,
                                       freeze_bn=freeze_bn)
        
        self.layer3 = self._make_layer(block_func=block, layer_id=3, 
                                       num_blocks=layers[2], 
                                       input_filters=filter_config[1][0], 
                                       output_filters=filter_config[2][0], 
                                       base_filters=int(filter_config[2][1]), 
                                       block_type=block_type,
                                       freeze_bn=freeze_bn)
        
        self.layer4 = self._make_layer(block_func=block, layer_id=4, 
                                       num_blocks=layers[3], 
                                       input_filters=filter_config[2][0], 
                                       output_filters=filter_config[3][0], 
                                       base_filters=int(filter_config[3][1]), 
                                       block_type=block_type,
                                       freeze_bn=freeze_bn)
        
        if cfg.SSL.ENABLE and cfg.SSL.MODE == "pretext":
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AvgPool3d((final_temporal_kernel, final_spatial_kernel, final_spatial_kernel), stride=(1, 1, 1), padding=(0, 0, 0))

            dropout_rate=cfg.MODEL.DROPOUT_RATE
            if dropout_rate > 0.0:
                self.dropout = nn.Dropout(dropout_rate)

            self.projection = nn.Linear(filter_config[3][0], num_classes, bias=True)

            # Softmax for evaluation and testing.
            act_func = cfg.MODEL.HEAD_ACT
            if act_func == "softmax":
                self.act = nn.Softmax(dim=4)
            elif act_func == "sigmoid":
                self.act = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    "{} is not supported as an activation"
                    "function.".format(act_func)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                if cfg.SSL.ENABLE:
                    continue
                else:
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
    
    def _make_layer(self, block_func, layer_id, num_blocks, 
                        input_filters, output_filters, base_filters,
                            block_type, freeze_bn):

        layers = []
        for idx in range(num_blocks):
            down_sampling = False
            if layer_id > 1 and idx == 0:
                down_sampling = True
            use_temp_conv = False
            if layer_id == 1 or (layer_id in [2, 3] and idx % 2 == 0) or (layer_id == 4 and idx % 2 == 1):
                use_temp_conv = True
            temporal_depth = int(self.sample_duration / pow(2, (layer_id - 1)))
            mgma_config = self._get_mgma_config(layer_id, idx, self.mgma_config)
            if len(mgma_config) == 0:
                layers.append(block_func[0](input_filters, output_filters, base_filters,
                                            use_temp_conv, self.se_ratio, down_sampling=down_sampling, 
                                                block_type=block_type, norm_module=self.norm_module,
                                                    freeze_bn=freeze_bn))
            else:
                layers.append(block_func[1](input_filters, output_filters, base_filters,
                                            mgma_config, use_temp_conv, down_sampling=down_sampling, 
                                                block_type=block_type, norm_module=self.norm_module,
                                                    freeze_bn=freeze_bn))
            input_filters = output_filters
        
        return nn.Sequential(*layers)

    def _get_mgma_config(self, layer_id, block_id, mgma_config):
        if len(mgma_config) == 0:
            return {}
        
        mgma_interval = mgma_config["INTERVAL"]
        use_first_unit = mgma_config["USE_FIRST_UNIT"]
        
        if layer_id in [2, 3]:
            if not use_first_unit and block_id == 0:
                return {}
            if mgma_interval > 1:
                if block_id % mgma_interval == 1:
                    return mgma_config
                else:
                    return {}
            else:
                return mgma_config
        else:
            return {}

    def forward(self, inputs):
        
        assert (
            len(inputs) == 1
        ), "Input tensor does not contain {} pathway".format(1)
        
        if self.model_name in ["r3d", "i3d"] or "_csn" in self.model_name:
            x = self.conv1(inputs[0])
            x = self.bn1(x)
            x = self.relu(x)
            if self.model_name == "i3d" or "_csn" in self.model_name:
                x = self.maxpool1(x)
        else:
            x = self.conv1_middle(inputs[0])
            x = self.bn1_middle(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.layer1(x)
        
        if self.model_name == "i3d":
            x = self.maxpool2(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not (self.cfg.SSL.ENABLE and self.cfg.SSL.MODE == "pretext"):
            # (N, C, T, H, W) -> (N, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))
            # Perform dropout.
            if hasattr(self, "dropout") and self.training:
                x = self.dropout(x)
            x = self.projection(x)
            
            # Performs fully convlutional inference.
            if not self.training:
                x = self.act(x)
                x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        
        return x
    
def obtain_arc(model_name, model_depth, sample_duration):
    assert model_depth in [18, 34, 50, 101, 152]
    
    if model_depth == 18:
        layers = (2, 2, 2, 2)
    elif model_depth == 34:
        layers = (3, 4, 6, 3)
    elif model_depth == 50:
        layers = (3, 4, 6, 3)
    elif model_depth == 101:
        layers = (3, 4, 23, 3)
    elif model_depth == 152:
        layers = (3, 8, 36, 3)
    
    if model_depth <= 18 or model_depth == 34:
        block = [BasicBlock, MGMABasicBlock]
    else:
        block = [Bottleneck, MGMABottleneck]
        
    if model_name == "r2plus1d":
        block_type = "2.5d"
    elif model_name == "r3d":
        block_type = "3d"
    elif model_name == "ir_csn":
        block_type = "3d-sep"
    elif model_name == "i3d":
        block_type = "i3d"
    
    if model_depth <= 34:
        filter_config = [
            [64, 64],
            [128, 128],
            [256, 256],
            [512, 512]
        ]
    else:
        filter_config = [
            [256, 64],
            [512, 128],
            [1024, 256],
            [2048, 512]
        ]

    if model_name == "i3d":
        final_temporal_kernel = int(sample_duration / 2)
    else:
        final_temporal_kernel = int(sample_duration / 8)
    final_spatial_kernel = 7
    
    return layers, block, block_type, filter_config, final_temporal_kernel, final_spatial_kernel
