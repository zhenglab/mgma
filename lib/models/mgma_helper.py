# Implementation for MGMA Module.

import torch.nn as nn
import numpy as np
import torch
import copy
import math

class MGMA(nn.Module):
    def __init__(self, input_filters, output_filters, mgma_config, freeze_bn=False, temporal_downsample=False, spatial_downsample=False):
        super(MGMA, self).__init__()
        
        self.freeze_bn = freeze_bn
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.mgma_type = mgma_config['TYPE'] if 'TYPE' in mgma_config else "TSA"
        self.mgma_num_groups = mgma_config['NUM_GROUPS'] if 'NUM_GROUPS' in mgma_config else 4
        
        layers = []
        if temporal_downsample and spatial_downsample:
            layers.append(self._make_downsample(kernel_type="U"))
        elif spatial_downsample:
            layers.append(self._make_downsample(kernel_type="S"))
        
        if self.input_filters % self.mgma_num_groups == 0:
            self.split_groups = [self.mgma_num_groups]
            self.split_channels = [self.input_filters]
        else:
            group_channels = math.ceil(self.input_filters / self.mgma_num_groups)
            split_groups_1 = self.input_filters // group_channels
            split_groups_2 = 1
            split_channels_1 = group_channels * split_groups_1
            split_channels_2 = self.input_filters - split_channels_1
            self.split_groups = [split_groups_1, split_groups_2]
            self.split_channels = [split_channels_1, split_channels_2]

        for i in range(len(self.split_groups)):
            if self.mgma_type in ["TA", "SA", "UA"]:
                if i == 0:
                    self.ma = nn.ModuleDict()
                layers_i = copy.deepcopy(layers)
                self._make_layers(layers_i, self.split_channels[i], self.split_channels[i], self.mgma_type[0], self.split_groups[i])
                self.ma[str(i)] = nn.Sequential(*layers_i)
            else:
                if i == 0:
                    self.ta = nn.ModuleDict()
                    self.sa = nn.ModuleDict()
                layers_t_i = copy.deepcopy(layers)
                layers_s_i = copy.deepcopy(layers)
                self._make_layers(layers_t_i, self.split_channels[i], self.split_channels[i], "T", self.split_groups[i])
                self._make_layers(layers_s_i, self.split_channels[i], self.split_channels[i], "S", self.split_groups[i])
                self.ta[str(i)] = nn.Sequential(*layers_t_i)
                self.sa[str(i)] = nn.Sequential(*layers_s_i)

    def _make_layers(self, layers, input_filters, output_filters, mgma_type, num_groups):
        layers.append(self._make_downsample(kernel_type=mgma_type))
        layers.append(self._make_interpolation(kernel_type=mgma_type))
        layers.append(self._make_conv_bn(input_filters, output_filters, kernel_type=mgma_type, groups=num_groups, use_relu=False))
        layers.append(self._make_activate())
    
    def _make_conv_bn(self, input_filters, out_filters, kernel_type="T", kernel=3, stride=1, padding=1, groups=1, use_bn=True, use_relu=True):
        layers = []
        
        if kernel_type.startswith("T"):
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(kernel, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), groups=groups, bias=False))
        elif kernel_type.startswith("S"):
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(1, kernel, kernel), stride=(1, stride, stride), padding=(0, padding, padding), groups=groups, bias=False))
        elif kernel_type.startswith("U"):    
            layers.append(nn.Conv3d(input_filters, out_filters, kernel_size=(kernel, kernel, kernel), stride=(stride, stride, stride), padding=(padding, padding, padding), groups=groups, bias=False))
        
        if use_bn:
            layers.append(nn.BatchNorm3d(out_filters, track_running_stats=(not self.freeze_bn)))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _make_downsample(self, kernel_type="T"):
        if kernel_type.startswith("T"):
            return nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        elif kernel_type.startswith("S"):
            return nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        elif kernel_type.startswith("U"):    
            return nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    
    def _make_interpolation(self, kernel_type="T"):
        if kernel_type.startswith("T"):
            self.upsample_scale_factor = (2, 1, 1)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
        elif kernel_type.startswith("S"):
            self.upsample_scale_factor = (1, 2, 2)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
        elif kernel_type.startswith("U"):
            self.upsample_scale_factor = (2, 2, 2)
            return nn.Upsample(scale_factor=self.upsample_scale_factor, mode='nearest')
    
    def _make_activate(self):
        return nn.Sigmoid()
        
    def forward(self, x):
        mgma_in = x
        
        if self.mgma_type in ["TA", "SA", "UA"]:
            ma_in_list = mgma_in.split(self.split_channels, 1)
            ma_out_list = []
            
            for i in range(len(ma_in_list)):
                ma_out_list.append(self.ma[str(i)](ma_in_list[i]))
            mgma_out = torch.cat(ma_out_list, 1)
            return mgma_out
        else:
            ta_in_list = mgma_in.split(self.split_channels, 1)
            ta_out_list = []
            
            for i in range(len(ta_in_list)):
                ta_out_list.append(self.ta[str(i)](ta_in_list[i]))
            mgma_ta_out = torch.cat(ta_out_list, 1)
            
            sa_in_list = mgma_in.split(self.split_channels, 1)
            sa_out_list = []
            
            for i in range(len(sa_in_list)):
                sa_out_list.append(self.sa[str(i)](sa_in_list[i]))
            mgma_sa_out = torch.cat(sa_out_list, 1)
            
            mgma_out = mgma_ta_out + mgma_sa_out
            return mgma_out
