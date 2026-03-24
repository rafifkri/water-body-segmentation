import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging


logger = logging.getLogger(__name__)


# ===========================
# ATTENTION MODULES
# ===========================
class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM)"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM)"""
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out) * x


class SCSeBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation Block"""
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SCSeBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) + self.spatial_attention(x)
        return out


# ===========================
# BASIC BUILDING BLOCKS
# ===========================
class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        normalization: str = 'batch',
        activation: str = 'relu'
    ):
        super(ConvBlock, self).__init__()
        
        # Normalization
        if normalization == 'batch':
            norm_layer = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            norm_layer = nn.InstanceNorm2d(out_channels)
        else:
            norm_layer = nn.Identity()
        
        # Activation
        if activation == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act_layer = nn.GELU()
        elif activation == 'swish':
            act_layer = nn.SiLU(inplace=True)
        else:
            act_layer = nn.Identity()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = norm_layer
        self.act1 = act_layer
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = norm_layer
        self.act2 = act_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        attention: bool = True,
        attention_type: str = 'scse'
    ):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Attention
        if attention:
            if attention_type == 'scse':
                self.attention = SCSeBlock(out_channels)
            elif attention_type == 'cbam':
                self.attention = nn.Sequential(
                    ChannelAttention(out_channels),
                    SpatialAttention()
                )
            else:
                self.attention = nn.Identity()
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.attention(x)
        return x


# ===========================
# U-NET ARCHITECTURE
# ===========================
class UNet(nn.Module):
    """
    U-Net model for semantic segmentation
    Reference: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        decoder_channels: Tuple[int, ...] = (512, 256, 128, 64, 32),
        dropout: float = 0.2,
        attention: bool = True,
        normalization: str = 'batch',
        activation: str = 'relu'
    ):
        super(UNet, self).__init__()
        
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Encoder
        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        
        for channels in encoder_channels:
            self.encoder.append(ConvBlock(prev_channels, channels, normalization=normalization, activation=activation))
            self.encoder.append(nn.MaxPool2d(2, 2))
            prev_channels = channels
        
        # Bottleneck (no pooling)
        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1] * 2, normalization=normalization, activation=activation)
        
        # Decoder
        self.decoder = nn.ModuleList()
        encoder_channels_list = list(encoder_channels)
        decoder_channels_list = list(decoder_channels)
        
        for i, out_ch in enumerate(decoder_channels_list):
            skip_ch = encoder_channels_list[len(encoder_channels_list) - i - 1]
            in_ch = encoder_channels_list[-1] * 2 if i == 0 else decoder_channels_list[i - 1]
            
            self.decoder.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout=dropout,
                    attention=attention
                )
            )
        
        # Output layer
        self.final_conv = nn.Conv2d(decoder_channels_list[-1], num_classes, kernel_size=1)
        self.out_activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Save ConvBlock output before MaxPool  
            if isinstance(layer, ConvBlock):
                encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        encoder_outputs.reverse()
        for i, decoder_block in enumerate(self.decoder):
            skip = encoder_outputs[i]
            x = decoder_block(x, skip)
        
        # Output
        x = self.final_conv(x)
        x = self.out_activation(x)
        
        return x


# ===========================
# SEGNET ARCHITECTURE
# ===========================
class SegNet(nn.Module):
    """
    SegNet model for semantic segmentation
    Reference: https://arxiv.org/abs/1511.02674
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 512),
        normalization: str = 'batch'
    ):
        super(SegNet, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        
        for channels in encoder_channels:
            self.encoder.append(nn.Sequential(
                ConvBlock(prev_channels, channels, normalization=normalization),
                nn.MaxPool2d(2, 2, return_indices=True)
            ))
            prev_channels = channels
        
        # Decoder
        self.decoder = nn.ModuleList()
        for channels in reversed(list(encoder_channels[:-1])) + [encoder_channels[0]]:
            self.decoder.append(nn.Sequential(
                nn.MaxUnpool2d(2, 2),
                ConvBlock(prev_channels, channels, normalization=normalization)
            ))
            prev_channels = channels
        
        # Output layer
        self.final_conv = nn.Conv2d(prev_channels, num_classes, kernel_size=1)
        self.out_activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_list = []
        sizes_list = []
        
        # Encoder
        for encoder_block in self.encoder:
            sizes_list.append(x.size())
            for layer in encoder_block:
                if isinstance(layer, nn.MaxPool2d):
                    x, indices = layer(x)
                    indices_list.append(indices)
                else:
                    x = layer(x)
        
        # Decoder
        indices_list.reverse()
        for i, decoder_block in enumerate(self.decoder):
            for j, layer in enumerate(decoder_block):
                if isinstance(layer, nn.MaxUnpool2d):
                    x = layer(x, indices_list[i], output_size=sizes_list[-(i+1)])
                else:
                    x = layer(x)
        
        # Output
        x = self.final_conv(x)
        x = self.out_activation(x)
        
        return x


# ===========================
# MODEL FACTORY
# ===========================
def create_model(
    model_type: str = 'unet',
    in_channels: int = 3,
    num_classes: int = 1,
    encoder_name: Optional[str] = None,
    encoder_weights: Optional[str] = None,
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64, 32),
    dropout: float = 0.2,
    attention: bool = True,
    attention_type: str = 'scse',
    normalization: str = 'batch',
    activation: str = 'relu'
) -> nn.Module:

    if model_type == 'unet':
        model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_channels=(64, 128, 256, 512, 1024),
            decoder_channels=decoder_channels,
            dropout=dropout,
            attention=attention,
            normalization=normalization,
            activation=activation
        )
    elif model_type == 'segnet':
        model = SegNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_channels=(64, 128, 256, 512, 512),
            normalization=normalization
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model with {count_parameters(model):,} parameters")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 256, 256)) -> None:
    """Print model summary"""
    from torchsummary import summary
    try:
        summary(model, input_shape)
    except:
        logger.warning("Could not print model summary. Install torchsummary: pip install torchsummary")
        logger.info(f"Model parameters: {count_parameters(model):,}")
