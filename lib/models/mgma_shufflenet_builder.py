# Implementation for MGMA-X3D.
# 
# Code partially referenced from:
# https://github.com/megvii-model/ShuffleNet-Series

import torch
import torch.nn as nn
from lib.models.batchnorm_helper import get_norm
from lib.models.mgma_helper import MGMA
from .build import MODEL_REGISTRY

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, temporal_downsample, block_type, mgma_config, freeze_bn=False, norm_module=nn.BatchNorm3d):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp
        
        self.mgma_config = mgma_config

        if block_type == "2.5d":
            branch_main = [
                # pw
                nn.Conv3d(inp, mid_channels, kernel_size=(3, 1, 1), stride=(2 if temporal_downsample else 1, 1, 1), padding=(1, 0, 0), bias=False),
                norm_module(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(mid_channels, mid_channels, kernel_size=(1, ksize, ksize), stride=(1, stride, stride), padding=(0, pad, pad), groups=mid_channels, bias=False),
                norm_module(mid_channels),
                # pw-linear
                nn.Conv3d(mid_channels, outputs, 1, 1, 0, bias=False),
                norm_module(outputs),
                nn.ReLU(inplace=True),
            ]
        else:
            branch_main = [
                # pw
                nn.Conv3d(inp, mid_channels, 1, 1, 0, bias=False),
                norm_module(mid_channels),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, ksize, ksize), stride=(2 if temporal_downsample else 1, stride, stride), padding=(1, pad, pad), groups=mid_channels, bias=False),
                norm_module(mid_channels),
                # pw-linear
                nn.Conv3d(mid_channels, outputs, 1, 1, 0, bias=False),
                norm_module(outputs),
                nn.ReLU(inplace=True),
            ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            if block_type == "2.5d":
                branch_proj = [
                    # dw
                    nn.Conv3d(inp, inp, kernel_size=(1, ksize, ksize), stride=(1, stride, stride), padding=(0, pad, pad), groups=inp, bias=False),
                    norm_module(inp),
                    # pw-linear
                    nn.Conv3d(inp, inp, kernel_size=(3, 1, 1), stride=(2 if temporal_downsample else 1, 1, 1), padding=(1, 0, 0), bias=False),
                    norm_module(inp),
                    nn.ReLU(inplace=True),
                ]
            else:
                branch_proj = [
                    # dw
                    nn.Conv3d(inp, inp, kernel_size=(3, ksize, ksize), stride=(2 if temporal_downsample else 1, stride, stride), padding=(1, pad, pad), groups=inp, bias=False),
                    norm_module(inp),
                    # pw-linear
                    nn.Conv3d(inp, inp, 1, 1, 0, bias=False),
                    norm_module(inp),
                    nn.ReLU(inplace=True),
                ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

        if len(mgma_config) > 0:
            input_filters = inp * 2 if self.stride == 1 else inp
            output_filters = oup
            self.mgma = MGMA(input_filters, output_filters, mgma_config, freeze_bn=freeze_bn, 
                             temporal_downsample=temporal_downsample, spatial_downsample=(stride == 2))
            self.mgma_use_shuffle = mgma_config['USE_SHUFFLE'] if 'USE_SHUFFLE' in mgma_config else True

    def forward(self, old_x):
        # shuffle
        if self.stride==1:
            if len(self.mgma_config) > 0:
                if self.mgma_use_shuffle:
                    x_proj, x = self.channel_shuffle(old_x)
                else:
                    x_proj, x = old_x.chunk(2, 1)
                x_mgma_in = torch.cat((x_proj, x), 1)
                x_block_out = torch.cat((x_proj, self.branch_main(x)), 1)
            else:
                x_proj, x = self.channel_shuffle(old_x)
                x_block_out = torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            if len(self.mgma_config) > 0:
                x_mgma_in = old_x
            x_block_out = torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        
        # output    
        if len(self.mgma_config) > 0:
            x_mgma_out = self.mgma(x_mgma_in)
            x_mgma_out = x_block_out * x_mgma_out
            x_block_out = x_block_out + x_mgma_out
            return x_block_out
        else:
            return x_block_out

    def channel_shuffle(self, x):
        batchsize, num_channels, temporal, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, temporal * height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, temporal, height, width)
        return x[0], x[1]

@MODEL_REGISTRY.register()
class MGMAShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(MGMAShuffleNet, self).__init__()
        
        sample_duration = cfg.DATA.NUM_FRAMES
        freeze_bn = cfg.BN.FREEZE_PARAMS
        num_classes = cfg.MODEL.NUM_CLASSES
        model_size = "2.0x"
        block_type = "3d"

        self.norm_module = get_norm(cfg)
        self.cfg = cfg
        
        if cfg.MGMA.TYPE != "NONE":
            self.mgma_config = cfg.MGMA
        else:
            self.mgma_config = {}
        
        final_temporal_kernel = int(sample_duration / 8)    
        final_spatial_kernel = 7

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv3d(3, input_channel, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            self.norm_module(input_channel),
            nn.ReLU(inplace=True),
        )

#         self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                mgma_block_config = self._get_mgma_config(idxstage, i, self.mgma_config)

                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, mid_channels=output_channel // 2, 
                                            ksize=3, stride=2, temporal_downsample=True, block_type=block_type, 
                                            mgma_config=mgma_block_config, freeze_bn=freeze_bn, norm_module=self.norm_module))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, mid_channels=output_channel // 2, 
                                            ksize=3, stride=1, temporal_downsample=False, block_type=block_type, 
                                            mgma_config=mgma_block_config, freeze_bn=freeze_bn, norm_module=self.norm_module))

                input_channel = output_channel
                
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv3d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            self.norm_module(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        if cfg.SSL.ENABLE and cfg.SSL.MODE == "pretext":
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AvgPool3d((final_temporal_kernel, final_spatial_kernel, final_spatial_kernel), stride=(1, 1, 1), padding=(0, 0, 0))

            dropout_rate=cfg.MODEL.DROPOUT_RATE
            if dropout_rate > 0.0 and self.model_size == '2.0x':
                self.dropout = nn.Dropout(dropout_rate)

            self.projection = nn.Linear(self.stage_out_channels[-1], num_classes, bias=True)

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
    
    def _get_mgma_config(self, layer_id, block_id, mgma_config):
        if len(mgma_config) == 0:
            return {}
        
        mgma_interval = mgma_config["INTERVAL"]
        use_first_unit = mgma_config["USE_FIRST_UNIT"]
        
        if layer_id in [0, 1]:
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
        
        x = self.first_conv(inputs[0])
#         x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        
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
