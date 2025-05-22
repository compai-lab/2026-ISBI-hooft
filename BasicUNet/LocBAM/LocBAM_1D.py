"""
# Copyright 2025 Donnate Bridget Hooft
#  Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inspired by the HANet architecture:
# https://github.com/shachoi/HANet 

# Implemented independently within the MONAI framework.
"""



# LocBAM: Lightweight Axial Attention Mechanism for 3D Medical Imaging
#
# This module implements **LocBAM (Local Axial Bottleneck Attention Module)** for 3D UNet architectures, enabling 
# efficient region-specific attention across each spatial axis (X, Y, Z) independently. Unlike standard self-attention 
# mechanisms, LocBAM avoids full 3D attention overhead by isolating and processing attention along one axis at a time.
#
# Key Features:
# - **Positional Encoding**: Injects axis-specific positional information using average or max pooling along orthogonal axes.
# - **Modular Attention Blocks**: Each of `LocBAM_Conv3D_{X,Y,Z}` performs 1D convolution-based attention across a single axis.
# - **Memory Efficiency**: By focusing attention computation along a single axis, memory use is significantly reduced 
#   compared to full 3D attention methods.
# - **Flexible Injection Points**: LocBAM can be inserted at various locations in a UNet pipeline (early, late, or both).
# - **Coordinate-aware Input**: Each model variation accepts an additional three channels encoding spatial (X, Y, Z) coordinates.
#
# Available Architectures:
# - `BasicUnetLocBAMs1`: Applies LocBAM before UNet encoding begins (input-level attention).
# - `BasicUnetLocBAMs2`: Applies LocBAM after the first convolutional block.
# - `BasicUnetLocBAMs3`: Applies LocBAM after the decoder's final upsampling layer (late fusion).
# - `BasicUnetLocBAMs4`: Applies LocBAM after the final output prediction (post-UNet).
# - `BasicUnetLocBAMs2_4`: Combines LocBAM at both early and late stages of the network.
#
# Each LocBAM variation includes:
# - Axis-specific attention (X, Y, Z) modules.
# - Optional positional injection at one or two stages.
# - Configurable number of layers, dropout, pooling method, and channel compression ratio (r_factor).
#
# Example Use Cases:
# --------------------------------------------------------------------------------------------
# • 3D medical image segmentation with spatially variant targets (e.g., organs or tumors).
# • Tasks where anatomical position plays a crucial role in interpretation.
import os
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet
import torch.nn.functional as F
import math
from config import Norm2d, Upsample, cfg   # check the directory
from monai.networks.blocks import Convolution



class PosEncoding1D_Z(nn.Module):
    def __init__(self, pooling = 'mean'):
        """Initialize the positional encoding module."""
        super(PosEncoding1D_Z, self).__init__()
        self.pooling = pooling
        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((1, 1, None))  # Pool over W and H, keep D
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((1, 1, None))  # Pool over W and H, keep D

    def forward(self, x, z_pos):
        """
        Inject positional encoding into the input feature map.

        Args:
            x (torch.Tensor): Input feature map (B x C x W x H x D).
            z_pos (torch.Tensor): Positional tensor (B x 1 x W x H x D).

        Returns:
            torch.Tensor: Feature map with positional encoding injected.
        """
       
        z_pos = self.rowpool(z_pos).squeeze(3).squeeze(2)
        return x + z_pos

class LocBAM_Conv3D_Z(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(LocBAM_Conv3D_Z, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()
        
 
        
        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                    kernel_size=1, stride=1, padding=0, bias=False),
                Norm2d(mid_1_channel),
                nn.ReLU(inplace=True))    

        if layer == 2:
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                        kernel_size=3, stride=1, padding=1, bias=True),
                    Norm2d(mid_2_channel),
                    nn.ReLU(inplace=True))    
            self.attention_third = nn.Sequential(
                    nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))

        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((1, 1, None))  # Pool over W and H, keep D
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((1, 1, None))  # Pool over W and H, keep D

        if pos_rfactor > 0:
            if is_encoding == 1:  # Use the positional encoding class
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D_Z(pooling = 'mean')
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D_Z(pooling = 'mean')
            else:
                raise ValueError("Unsupported positional encoding configuration.")

    def forward(self, x, out, z, return_attention=False, return_posmap=False, attention_loss=False):
        """
        Inputs:
            x : Input feature maps (B X C X W X H X D)
            out : Additional input (e.g., intermediate feature maps)
            z : Height tensor (B X 1 X W X H X D) representing positional information
        Returns:
            out : Self-attention value + input feature
        """
        B, C, W, H, D = x.shape  # Extract dimensions from Z
        
        # Pool over W and H dimensions
        x = self.rowpool(x).squeeze(3).squeeze(2)
        # print(f"x shape after squeezing: {x.shape}")
        # Inject positional encoding using z_pos
        if self.pos_injection == 1:
            x = self.pos_emb1d_1st(x, z)
            # print(f"x shape after pos_emb1d_1st: {x.shape}")

        if self.dropout_prob > 0.0:
            x = self.dropout(x)
            # print(f"x shape after dropout: {x.shape}")

        x = self.attention_first(x)
        # print(f"x shape after attention_first: {x.shape}")

        # Apply second-level positional injection if enabled
        if self.pos_injection == 2:
            x = self.pos_emb1d_2nd(x, z)
            # print(f"x shape after pos_emb1d_2nd: {x.shape}")

        x = self.attention_second(x)
        # print(f"x shape after attention_second: {x.shape}")

        if self.layer == 3:
            x = self.attention_third(x)
            # print(f"x shape after attention_third: {x.shape}")
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)
        else:
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)

        # print(f"x shape after sigmoid: {x.shape}")
        x = F.interpolate(x, size=D, mode='linear')
        # print(f"x shape after interpolate: {x.shape}")
        x_expanded = x.unsqueeze(2).unsqueeze(3)
        # print(f"x shape after unsq: {x_expanded.shape}")
        # print(f"out shape b4 mul: {out.shape}")
        out = torch.mul(out, x_expanded)
        # print(f"out shape after mul: {out.shape}")

        if return_attention:
            return out, x
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out


class PosEncoding1D_X(nn.Module):
    def __init__(self, pooling='mean'):
        """Initialize the positional encoding module."""
        super(PosEncoding1D_X, self).__init__()
        self.pooling = pooling
        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool over H and D, keep W
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((None, 1, 1))  # Pool over H and D, keep W

    def forward(self, x, z_pos):
        """
        Inject positional encoding into the input feature map.

        Args:
            x (torch.Tensor): Input feature map (B x C x W x H x D).
            z_pos (torch.Tensor): Positional tensor (B x 1 x W x H x D).

        Returns:
            torch.Tensor: Feature map with positional encoding injected.
        """
        # Pool over H and D dimensions and keep W
        z_pos = self.rowpool(z_pos).squeeze(4).squeeze(3)  # Now it's (B, 1, W)
        return x + z_pos


class LocBAM_Conv3D_X(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(LocBAM_Conv3D_X, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                    kernel_size=1, stride=1, padding=0, bias=False),
                Norm2d(mid_1_channel),
                nn.ReLU(inplace=True))    

        if layer == 2:
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                        kernel_size=3, stride=1, padding=1, bias=True),
                    Norm2d(mid_2_channel),
                    nn.ReLU(inplace=True))    
            self.attention_third = nn.Sequential(
                    nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))

        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool over H and D, keep W
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((None, 1, 1))  # Pool over H and D, keep W

        if pos_rfactor > 0:
            if is_encoding == 1:  # Use the positional encoding class
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D_X(pooling='mean')
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D_X(pooling='mean')
            else:
                raise ValueError("Unsupported positional encoding configuration.")

    def forward(self, x, out, z, return_attention=False, return_posmap=False, attention_loss=False):
        """
        Inputs:
            x : Input feature maps (B X C X W X H X D)
            out : Additional input (e.g., intermediate feature maps)
            z : Height tensor (B X 1 X W X H X D) representing positional information
        Returns:
            out : Self-attention value + input feature
        """
        B, C, W, H, D = x.shape  # Extract dimensions from Z
        # Process height tensor Z to create z_pos
        x = self.rowpool(x).squeeze(4).squeeze(3)  # Now x has shape (B, C, W)
        # print(f"x shape after squeezing: {x.shape}")

        # Inject positional encoding using z_pos
        if self.pos_injection == 1:
            x = self.pos_emb1d_1st(x, z)
            # print(f"x shape after pos_emb1d_1st: {x.shape}")

        if self.dropout_prob > 0.0:
            x = self.dropout(x)
            # print(f"x shape after dropout: {x.shape}")

        x = self.attention_first(x)
        # print(f"x shape after attention_first: {x.shape}")

        # Apply second-level positional injection if enabled
        if self.pos_injection == 2:
            x = self.pos_emb1d_2nd(x, z)
            # print(f"x shape after pos_emb1d_2nd: {x.shape}")

        x = self.attention_second(x)
        # print(f"x shape after attention_second: {x.shape}")

        if self.layer == 3:
            x = self.attention_third(x)
            # print(f"x shape after attention_third: {x.shape}")
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)
        else:
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)

        # print(f"x shape after sigmoid: {x.shape}")
        x = F.interpolate(x, size=W, mode='linear')  # Now interpolating along the W dimension
        # print(f"x shape after interpolate: {x.shape}")

        # Expanding the tensor to match the number of dimensions needed for multiplication
        x_expanded = x.unsqueeze(3).unsqueeze(4)
        # print(f"x shape after unsqueeze: {x_expanded.shape}")
        # print(f"out shape before mul: {out.shape}")

        # Perform element-wise multiplication between out and x_expanded
        out = torch.mul(out, x_expanded)
        # print(f"out shape after mul: {out.shape}")

        if return_attention:
            return out, x
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out
# Ensure LocBAM_Conv3D_Z and PosEncoding1D_Z are defined as provided earlier
class PosEncoding1D_Y(nn.Module):
    def __init__(self, pooling='mean'):
        """Initialize the positional encoding module."""
        super(PosEncoding1D_Y, self).__init__()
        self.pooling = pooling
        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((1, None, 1))  # Pool over W and D, keep H
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((1, None, 1))  # Pool over W and D, keep H

    def forward(self, x, z_pos):
        """
        Inject positional encoding into the input feature map.

        Args:
            x (torch.Tensor): Input feature map (B x C x W x H x D).
            z_pos (torch.Tensor): Positional tensor (B x 1 x W x H x D).

        Returns:
            torch.Tensor: Feature map with positional encoding injected.
        """
        # Pool over W and D dimensions, keep H
        z_pos = self.rowpool(z_pos).squeeze(4).squeeze(2)  # Now it's (B, 1, H)
        return x + z_pos

class LocBAM_Conv3D_Y(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(LocBAM_Conv3D_Y, self).__init__()

        self.pooling = pooling
        self.pos_injection = pos_injection
        self.layer = layer
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        self.attention_first = nn.Sequential(
                nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                    kernel_size=1, stride=1, padding=0, bias=False),
                Norm2d(mid_1_channel),
                nn.ReLU(inplace=True))    

        if layer == 2:
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)
            self.attention_second = nn.Sequential(
                    nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                        kernel_size=3, stride=1, padding=1, bias=True),
                    Norm2d(mid_2_channel),
                    nn.ReLU(inplace=True))    
            self.attention_third = nn.Sequential(
                    nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                        kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True))

        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool3d((1, None, 1))  # Pool over W and D, keep H
        else:
            self.rowpool = nn.AdaptiveMaxPool3d((1, None, 1))  # Pool over W and D, keep H

        if pos_rfactor > 0:
            if is_encoding == 1:  # Use the positional encoding class
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D_Y(pooling='mean')
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D_Y(pooling='mean')
            else:
                raise ValueError("Unsupported positional encoding configuration.")

    def forward(self, x, out, z, return_attention=False, return_posmap=False, attention_loss=False):
        """
        Inputs:
            x : Input feature maps (B X C X W X H X D)
            out : Additional input (e.g., intermediate feature maps)
            z : Height tensor (B X 1 X W X H X D) representing positional information
        Returns:
            out : Self-attention value + input feature
        """
        B, C, W, H, D = x.shape  # Extract dimensions from Z

        # Process height tensor Z to create z_pos
        x = self.rowpool(x).squeeze(4).squeeze(2)  # Now x has shape (B, C, H)
        # print(f"x shape after squeezing: {x.shape}")

        # Inject positional encoding using z_pos
        if self.pos_injection == 1:
            x = self.pos_emb1d_1st(x, z)
            # print(f"x shape after pos_emb1d_1st: {x.shape}")

        if self.dropout_prob > 0.0:
            x = self.dropout(x)
            # print(f"x shape after dropout: {x.shape}")

        x = self.attention_first(x)
        # print(f"x shape after attention_first: {x.shape}")

        # Apply second-level positional injection if enabled
        if self.pos_injection == 2:
            x = self.pos_emb1d_2nd(x, z)
            # print(f"x shape after pos_emb1d_2nd: {x.shape}")

        x = self.attention_second(x)
        # print(f"x shape after attention_second: {x.shape}")

        if self.layer == 3:
            x = self.attention_third(x)
            # print(f"x shape after attention_third: {x.shape}")
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)
        else:
            if attention_loss:
                last_attention = x
            x = self.sigmoid(x)

        # print(f"x shape after sigmoid: {x.shape}")
        x = F.interpolate(x, size=H, mode='linear')  # Now interpolating along the H dimension
        # print(f"x shape after interpolate: {x.shape}")

        # Expanding the tensor to match the number of dimensions needed for multiplication
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        # print(f"x shape after unsqueeze: {x_expanded.shape}")
        # print(f"out shape before mul: {out.shape}")

        # Perform element-wise multiplication between out and x_expanded
        out = torch.mul(out, x_expanded)
        # print(f"out shape after mul: {out.shape}")

        if return_attention:
            return out, x
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out

# This was the best working model, with LocBAM in its second place (After 1st convolution in UNet)

class BasicUnetLocBAMs2(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=4,  # Input tensor channels
        out_channels=14,  # Output tensor channels
        features=(32, 32, 64, 128, 256, 32),  # UNet feature sizes
        dropout=0.0,
        LocBAM_params=None  # LocBAM parameters
    ):
        super(BasicUnetLocBAMs2, self).__init__()

        # Base UNet from MONAI
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )

        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params = {
                "in_channel": features[0],  # Match first UNet layer's feature size
                "out_channel": features[0],  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y = LocBAM_Conv3D_Y(**LocBAM_params)
        self.LocBAM_x = LocBAM_Conv3D_X(**LocBAM_params)
        self.LocBAM_z = LocBAM_Conv3D_Z(**LocBAM_params)

       
        # Final convolution layer
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=features[-1],  # UNet output channels
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None
        )
        # Fusion layer (can be customized)
        self.fusion_layer = nn.Conv3d(features[0]* 3, features[0], kernel_size=1)

    def combine_LocBAM_outputs(self, height, width, depth):
        return torch.cat([height, width, depth], dim=1)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, W, H, D). The channels are:
                - Channel 0: Main input (x)
                - Channel 1: Width tensor (xi)
                - Channel 2: Depth tensor (y)
                - Channel 3: Height tensor (z)

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        # # print(f"[Input Tensor Shape] {x.shape}")
        
        # Extract input coordinates
        input_coords = x[:, 1:, ...]  # Extract the coordinate channels
        x_main = x[:, :1, ...]        # Extract the main input channel
        # # print(f"[Main Input Shape] {x_main.shape}")
        # # print(f"[Input Coordinates Shape] {input_coords.shape}")

        # Pass through the first convolution block
        x0 = self.unet.conv_0(x_main)
        # # print(f"[After conv_0 Shape] {x0.shape}")
        
        LocBAM_out_height = self.LocBAM_x(x0, x0, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width = self.LocBAM_y(x0, x0, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth = self.LocBAM_z(x0, x0, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        combined = self.combine_LocBAM_outputs(LocBAM_out_height, LocBAM_out_width, LocBAM_out_depth)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output = self.fusion_layer(combined)
        # combined_LocBAM = LocBAM_out_height + LocBAM_out_width + LocBAM_out_depth
        # combined_LocBAM = LocBAM_out_height * LocBAM_out_width * LocBAM_out_depth
        # # print(f"[Fused Shape] {fused_output.shape}")

        # Pass through the remaining UNet layers
        x1 = self.unet.down_1(fused_output)
        # # print(f"[After down_1 Shape] {x1.shape}")
        x2 = self.unet.down_2(x1)
        # # print(f"[After down_2 Shape] {x2.shape}")
        x3 = self.unet.down_3(x2)
        # # print(f"[After down_3 Shape] {x3.shape}")
        x4 = self.unet.down_4(x3)
        # # print(f"[After down_4 Shape] {x4.shape}")

        # Upsampling blocks
        u4 = self.unet.upcat_4(x4, x3)
        # # print(f"[After upcat_4 Shape] {u4.shape}")
        u3 = self.unet.upcat_3(u4, x2)
        # # print(f"[After upcat_3 Shape] {u3.shape}")
        u2 = self.unet.upcat_2(u3, x1)
        # # print(f"[After upcat_2 Shape] {u2.shape}")
        u1 = self.unet.upcat_1(u2, fused_output)
        # # print(f"[After upcat_1 Shape] {u1.shape}")

        # Final convolution
        output = self.final_conv(u1)
        # # print(f"[Final Output Shape] {output.shape}")

        return output


# Below are examples of where LocBAM can be used in alternative palces within a UNet architecture
# Feel free to try out various model types to see which one performs best for your designated task

class BasicUnetLocBAMs4(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,  # Input tensor channels
        out_channels=14,  # Output tensor channels
        features=(32, 32, 64, 128, 256, 32),  # UNet feature sizes
        dropout=0.0,
        LocBAM_params=None  # LocBAM parameters
    ):
        super(BasicUnetLocBAMs4, self).__init__()

        # Base UNet from MONAI
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )

        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params = {
                "in_channel": out_channels,  # Match first UNet layer's feature size
                "out_channel": out_channels,  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y = LocBAM_Conv3D_Y(**LocBAM_params)
        self.LocBAM_x = LocBAM_Conv3D_X(**LocBAM_params)
        self.LocBAM_z = LocBAM_Conv3D_Z(**LocBAM_params)

        # Final convolution layer
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=features[-1] ,#+ 3,  # 3 channels for coordinates
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None
        )
        self.fusion_layer = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def combine_LocBAM_outputs(self, height, width, depth):
        return torch.cat([height, width, depth], dim=1)

    def forward(self, x):
        # # print(f"[Input Tensor Shape] {x.shape}")
        
        # Extract input coordinates
        input_coords = x[:, 1:, ...]  # Extract the coordinate channels
        x_main = x[:, :1, ...]        # Extract the main input channel
        # # print(f"[Main Input Shape] {x_main.shape}")
        # # print(f"[Input Coordinates Shape] {input_coords.shape}")

        # Pass through the first convolution block
        x0 = self.unet.conv_0(x_main)
        # # print(f"[After conv_0 Shape] {x0.shape}")
        
        # Downsampling blocks
        x1 = self.unet.down_1(x0)
        # # print(f"[After down_1 Shape] {x1.shape}")
        x2 = self.unet.down_2(x1)
        # # print(f"[After down_2 Shape] {x2.shape}")
        x3 = self.unet.down_3(x2)
        # # print(f"[After down_3 Shape] {x3.shape}")
        x4 = self.unet.down_4(x3)
        # # print(f"[After down_4 Shape] {x4.shape}")

        # Upsampling blocks
        u4 = self.unet.upcat_4(x4, x3)
        # # print(f"[After upcat_4 Shape] {u4.shape}")
        u3 = self.unet.upcat_3(u4, x2)
        # # print(f"[After upcat_3 Shape] {u3.shape}")
        u2 = self.unet.upcat_2(u3, x1)
        # # print(f"[After upcat_2 Shape] {u2.shape}")
        u1 = self.unet.upcat_1(u2, x0)
        # # print(f"[After upcat_1 Shape] {u1.shape}")

        # Final convolution
        output_1 = self.final_conv(u1)
            # Apply LocBAM modules
        LocBAM_out_height = self.LocBAM_x(output_1, output_1, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width = self.LocBAM_y(output_1, output_1, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth = self.LocBAM_z(output_1, output_1, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        combined = self.combine_LocBAM_outputs(LocBAM_out_height, LocBAM_out_width, LocBAM_out_depth)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output = self.fusion_layer(combined)
        # combined_LocBAM = LocBAM_out_height + LocBAM_out_width + LocBAM_out_depth
        # combined_LocBAM = LocBAM_out_height * LocBAM_out_width * LocBAM_out_depth
        # # print(f"[Fused Shape] {fused_output.shape}")
             

        
        # # print(f"[Final Output Shape] {output.shape}")

        return fused_output

class BasicUnetLocBAMs2_4(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,  # Input tensor channels
        out_channels=14,  # Output tensor channels
        features=(32, 32, 64, 128, 256, 32),  # UNet feature sizes
        dropout=0.0,
        LocBAM_params=None  # LocBAM parameters
    ):
        super(BasicUnetLocBAMs2_4, self).__init__()

        # Base UNet from MONAI
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )
        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params_1 = {
                "in_channel": features[0],  # Match first UNet layer's feature size
                "out_channel": features[0],  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y_1 = LocBAM_Conv3D_Y(**LocBAM_params_1)
        self.LocBAM_x_1 = LocBAM_Conv3D_X(**LocBAM_params_1)
        self.LocBAM_z_1 = LocBAM_Conv3D_Z(**LocBAM_params_1)

       
        
        # Fusion layer (can be customized)
        self.fusion_layer_1 = nn.Conv3d(features[0]* 3, features[0], kernel_size=1)
        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params = {
                "in_channel": out_channels,  # Match first UNet layer's feature size
                "out_channel": out_channels,  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y = LocBAM_Conv3D_Y(**LocBAM_params)
        self.LocBAM_x = LocBAM_Conv3D_X(**LocBAM_params)
        self.LocBAM_z = LocBAM_Conv3D_Z(**LocBAM_params)

        # Final convolution layer
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=features[-1] ,#+ 3,  # 3 channels for coordinates
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None
        )
        self.fusion_layer = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def combine_LocBAM_outputs(self, height, width, depth):
        return torch.cat([height, width, depth], dim=1)

    def forward(self, x):
        # # print(f"[Input Tensor Shape] {x.shape}")
        
        # Extract input coordinates
        input_coords = x[:, 1:, ...]  # Extract the coordinate channels
        x_main = x[:, :1, ...]        # Extract the main input channel
        # # print(f"[Main Input Shape] {x_main.shape}")
        # # print(f"[Input Coordinates Shape] {input_coords.shape}")

        # Pass through the first convolution block
        x0 = self.unet.conv_0(x_main)
        # # print(f"[After conv_0 Shape] {x0.shape}")
        LocBAM_out_height_1 = self.LocBAM_x_1(x0, x0, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width_1 = self.LocBAM_y_1(x0, x0, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth_1 = self.LocBAM_z_1(x0, x0, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        combined_1 = self.combine_LocBAM_outputs(LocBAM_out_height_1, LocBAM_out_width_1, LocBAM_out_depth_1)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output_1 = self.fusion_layer_1(combined_1)
        # Downsampling blocks
        x1 = self.unet.down_1(fused_output_1)
        # # print(f"[After down_1 Shape] {x1.shape}")
        x2 = self.unet.down_2(x1)
        # # print(f"[After down_2 Shape] {x2.shape}")
        x3 = self.unet.down_3(x2)
        # # print(f"[After down_3 Shape] {x3.shape}")
        x4 = self.unet.down_4(x3)
        # # print(f"[After down_4 Shape] {x4.shape}")

        # Upsampling blocks
        u4 = self.unet.upcat_4(x4, x3)
        # # print(f"[After upcat_4 Shape] {u4.shape}")
        u3 = self.unet.upcat_3(u4, x2)
        # # print(f"[After upcat_3 Shape] {u3.shape}")
        u2 = self.unet.upcat_2(u3, x1)
        # # print(f"[After upcat_2 Shape] {u2.shape}")
        u1 = self.unet.upcat_1(u2, fused_output_1)
        # # print(f"[After upcat_1 Shape] {u1.shape}")

        # Final convolution
        output_1 = self.final_conv(u1)
            # Apply LocBAM modules
        LocBAM_out_height = self.LocBAM_x(output_1, output_1, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width = self.LocBAM_y(output_1, output_1, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth = self.LocBAM_z(output_1, output_1, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        combined = self.combine_LocBAM_outputs(LocBAM_out_height, LocBAM_out_width, LocBAM_out_depth)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output = self.fusion_layer(combined)
        # combined_LocBAM = LocBAM_out_height + LocBAM_out_width + LocBAM_out_depth
        # combined_LocBAM = LocBAM_out_height * LocBAM_out_width * LocBAM_out_depth
        # # print(f"[Fused Shape] {fused_output.shape}")
             

        
        # # print(f"[Final Output Shape] {output.shape}")

        return fused_output

class BasicUnetLocBAMs1(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=4,  # Input tensor channels
        out_channels=14,  # Output tensor channels
        features=(32, 32, 64, 128, 256, 32),  # UNet feature sizes
        dropout=0.0,
        LocBAM_params=None  # LocBAM parameters
    ):
        super(BasicUnetLocBAMs1, self).__init__()

        # Base UNet from MONAI
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )

        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params = {
                "in_channel": 1,  # Match first UNet layer's feature size
                "out_channel": 1,  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y = LocBAM_Conv3D_Y(**LocBAM_params)
        self.LocBAM_x = LocBAM_Conv3D_X(**LocBAM_params)
        self.LocBAM_z = LocBAM_Conv3D_Z(**LocBAM_params)


        # Final convolution layer
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=features[-1],  # UNet output channels
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None
        )
        # Fusion layer (can be customized)
        self.fusion_layer = nn.Conv3d(1 * 3, 1, kernel_size=1)

    def combine_LocBAM_outputs(self, height, width, depth):
        return torch.cat([height, width, depth], dim=1)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, W, H, D). The channels are:
                - Channel 0: Main input (x)
                - Channel 1: Width tensor (xi)
                - Channel 2: Depth tensor (y)
                - Channel 3: Height tensor (z)

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        # # print(f"[Input Tensor Shape] {x.shape}")
        
        # Extract input coordinates
        input_coords = x[:, 1:, ...]  # Extract the coordinate channels
        x_main = x[:, :1, ...]        # Extract the main input channel
        # # print(f"[Main Input Shape] {x_main.shape}")
        # # print(f"[Input Coordinates Shape] {input_coords.shape}")

                       # Apply LocBAM modules
        LocBAM_out_height = self.LocBAM_x(x_main, x_main, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width = self.LocBAM_y(x_main, x_main, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth = self.LocBAM_z(x_main, x_main, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        # Concatenate outputs along the channel dimension
        combined = self.combine_LocBAM_outputs(LocBAM_out_height, LocBAM_out_width, LocBAM_out_depth)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output = self.fusion_layer(combined)
        # combined_LocBAM = LocBAM_out_height + LocBAM_out_width + LocBAM_out_depth
        # combined_LocBAM = LocBAM_out_height * LocBAM_out_width * LocBAM_out_depth
        # # print(f"[Fused Shape] {fused_output.shape}")
        x0 = self.unet.conv_0(fused_output)
        
        # Pass through the remaining UNet layers
        x1 = self.unet.down_1(x0)
        # # print(f"[After down_1 Shape] {x1.shape}")
        x2 = self.unet.down_2(x1)
        # # print(f"[After down_2 Shape] {x2.shape}")
        x3 = self.unet.down_3(x2)
        # # print(f"[After down_3 Shape] {x3.shape}")
        x4 = self.unet.down_4(x3)
        # # print(f"[After down_4 Shape] {x4.shape}")

        # Upsampling blocks
        u4 = self.unet.upcat_4(x4, x3)
        # # print(f"[After upcat_4 Shape] {u4.shape}")
        u3 = self.unet.upcat_3(u4, x2)
        # # print(f"[After upcat_3 Shape] {u3.shape}")
        u2 = self.unet.upcat_2(u3, x1)
        # # print(f"[After upcat_2 Shape] {u2.shape}")
        u1 = self.unet.upcat_1(u2, x0)
        # # print(f"[After upcat_1 Shape] {u1.shape}")

        # Final convolution
        output = self.final_conv(u1)
        # # print(f"[Final Output Shape] {output.shape}")

        return output

class BasicUnetLocBAMs3(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,  # Input tensor channels
        out_channels=14,  # Output tensor channels
        features=(32, 32, 64, 128, 256, 32),  # UNet feature sizes
        dropout=0.0,
        LocBAM_params=None  # LocBAM parameters
    ):
        super(BasicUnetLocBAMs3, self).__init__()

        # Base UNet from MONAI
        self.unet = BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )

        # Define LocBAM parameters
        if LocBAM_params is None:
            LocBAM_params = {
                "in_channel": features[-1],  # Match first UNet layer's feature size
                "out_channel": features[-1],  # Match feature size
                "kernel_size": 3,
                "r_factor": 32,
                "layer": 3,
                "pos_injection": 1,
                "is_encoding": 1,
                "pos_rfactor": 8,
                "pooling": 'mean',
                "dropout_prob": dropout,
                "pos_noise": 0.0,
            }

        # Instantiate three LocBAM modules
        self.LocBAM_y = LocBAM_Conv3D_Y(**LocBAM_params)
        self.LocBAM_x = LocBAM_Conv3D_X(**LocBAM_params)
        self.LocBAM_z = LocBAM_Conv3D_Z(**LocBAM_params)

        # Final convolution layer
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=features[-1] ,#+ 3,  # 3 channels for coordinates
            out_channels=out_channels,
            kernel_size=1,
            act=None,
            norm=None
        )
        self.fusion_layer = nn.Conv3d(features[-1] * 3, features[-1], kernel_size=1)

    def combine_LocBAM_outputs(self, height, width, depth):
        return torch.cat([height, width, depth], dim=1)

    def forward(self, x):
        # # print(f"[Input Tensor Shape] {x.shape}")
        
        # Extract input coordinates
        input_coords = x[:, 1:, ...]  # Extract the coordinate channels
        x_main = x[:, :1, ...]        # Extract the main input channel
        # # print(f"[Main Input Shape] {x_main.shape}")
        # # print(f"[Input Coordinates Shape] {input_coords.shape}")

        # Pass through the first convolution block
        x0 = self.unet.conv_0(x_main)
        # # print(f"[After conv_0 Shape] {x0.shape}")
        
        # Downsampling blocks
        x1 = self.unet.down_1(x0)
        # # print(f"[After down_1 Shape] {x1.shape}")
        x2 = self.unet.down_2(x1)
        # # print(f"[After down_2 Shape] {x2.shape}")
        x3 = self.unet.down_3(x2)
        # # print(f"[After down_3 Shape] {x3.shape}")
        x4 = self.unet.down_4(x3)
        # # print(f"[After down_4 Shape] {x4.shape}")

        # Upsampling blocks
        u4 = self.unet.upcat_4(x4, x3)
        # # print(f"[After upcat_4 Shape] {u4.shape}")
        u3 = self.unet.upcat_3(u4, x2)
        # # print(f"[After upcat_3 Shape] {u3.shape}")
        u2 = self.unet.upcat_2(u3, x1)
        # # print(f"[After upcat_2 Shape] {u2.shape}")
        u1 = self.unet.upcat_1(u2, x0)
        # # print(f"[After upcat_1 Shape] {u1.shape}")

            # Apply LocBAM modules
        LocBAM_out_height = self.LocBAM_x(u1, u1, input_coords[:, :1, ...])
        # # print(f"[LocBAM Height Output Shape] {LocBAM_out_height.shape}")
        LocBAM_out_width = self.LocBAM_y(u1, u1, input_coords[:, 1:2, ...])
        # # print(f"[LocBAM Width Output Shape] {LocBAM_out_width.shape}")
        LocBAM_out_depth = self.LocBAM_z(u1, u1, input_coords[:, 2:, ...])
        # # print(f"[LocBAM Depth Output Shape] {LocBAM_out_depth.shape}")

        # Combine LocBAM outputs
        combined = self.combine_LocBAM_outputs(LocBAM_out_height, LocBAM_out_width, LocBAM_out_depth)
        
        # # print(f"[combined Shape] {combined.shape}")
        # Fuse the combined outputs
        fused_output = self.fusion_layer(combined)
        # combined_LocBAM = LocBAM_out_height + LocBAM_out_width + LocBAM_out_depth
        # combined_LocBAM = LocBAM_out_height * LocBAM_out_width * LocBAM_out_depth
        # # print(f"[Fused Shape] {fused_output.shape}")
             

        # Final convolution
        output = self.final_conv(fused_output)
        # # print(f"[Final Output Shape] {output.shape}")

        return output


# Outcommented code below for testing model intilialization, used for debugging

# model = BasicUnetLocBAMs2(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=14,
#     features=(32, 32, 64, 128, 256, 32),
#     dropout=0.1,
#     LocBAM_params=None,
# )#.cuda()
 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# # Example inputs
# x = torch.randn(128, 4, 32, 32, 32).cuda()  # Input tensor (B, C, W, H, D)
# total_params = sum(p.numel() for p in model.parameters())
# print(total_params)
# output = model(x)
# print(output.shape)


