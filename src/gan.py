import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple
from typing import Tuple, List, Any

from .conv import ConfigurableCNN, ConvBlock, ConvTransposeBlock


def _validate_img_shape(img_shape: Tuple[int, int, int]) -> None:
    if not isinstance(img_shape, (tuple, list)) or len(img_shape) != 3:
        raise TypeError(f"img_shape must be a tuple/list of 3 ints (C, H, W), got: {img_shape!r}")
    if not all(isinstance(x, int) and x > 0 for x in img_shape):
        raise ValueError(f"img_shape elements must be positive integers, got: {img_shape!r}")


def _validate_conv_config(conv_layers_config: Any) -> None:
    if not isinstance(conv_layers_config, list) or len(conv_layers_config) == 0:
        raise TypeError("conv_layers_config must be a non-empty list of layer configuration dicts.")

class Discriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int], conv_layers_config: List[dict]):
        """
        Discriminator that maps an image of shape img_shape to a scalar probability.
        Performs validation of inputs and wraps conv output sizing in guarded blocks.
        """
        super(Discriminator, self).__init__()

        _validate_img_shape(img_shape)
        _validate_conv_config(conv_layers_config)

        self.img_shape = tuple(img_shape)

        try:
            self.cnn = ConfigurableCNN(layers_config=conv_layers_config)
        except Exception as e:
            raise RuntimeError(f"Failed to construct ConfigurableCNN for discriminator: {e}") from e

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            raise TypeError(f"Expected img to be a torch.Tensor, got {type(img)}")
        if img.dim() != 4:
            raise ValueError(f"Expected img tensor to have 4 dims (B, C, H, W), got shape {tuple(img.shape)}")

        # Basic shape compatibility check (C, H, W)
        if tuple(img.shape[1:]) != self.img_shape:
            raise ValueError(f"Input images must have shape (B, {self.img_shape[0]}, {self.img_shape[1]}, {self.img_shape[2]}), "
                             f"but got {tuple(img.shape)}")

        out = self.cnn(img)

        return out
    
class DenseNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(DenseNetEncoder, self).__init__()

        densenet = models.densenet121(
            weights=(models.DenseNet121_Weights.DEFAULT if pretrained else None)
        )
        self.features = densenet.features

        self.block1 = nn.Sequential(self.features.denseblock1, self.features.transition1)
        self.block2 = nn.Sequential(self.features.denseblock2, self.features.transition2)
        self.block3 = nn.Sequential(self.features.denseblock3, self.features.transition3)

class SideBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SideBranch, self).__init__()

        self.proj = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            activation="linear",
            use_batch_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ds = F.avg_pool2d(x, 2)
        return self.proj(x_ds)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, grow_channels: int, out_channels: int, upsample: bool = True):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample

        self.dense = self._make_dense(in_channels, grow_channels)

        self.up_trans = ConvTransposeBlock(
            in_channels=in_channels + grow_channels,
            out_channels=out_channels,
            kernel_size=1,  
            stride=1,       
            padding=0,      
            activation="relu",
            use_batch_norm=True
        )

    def _make_dense(self, in_c: int, grow_c: int):
        return nn.Sequential(
            ConvBlock(
                in_c,
                grow_c * 4,
                kernel_size=1,
                padding=0,
                activation="relu",
                use_batch_norm=True,
            ),
            ConvBlock(
                grow_c * 4,
                grow_c,
                kernel_size=3,
                padding=1,
                activation="relu",
                use_batch_norm=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_feat = self.dense(x)

        x_dense = torch.cat([x, dense_feat], dim=1)

        out = self.up_trans(x_dense)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        return out


class FDGANGenerator(nn.Module):
    def __init__(self, output_same_size=True):
        super(FDGANGenerator, self).__init__()
        self.output_same_size = output_same_size

        self.encoder = DenseNetEncoder()

        self.conv_in = ConvBlock(3, 64, kernel_size=3, padding=1, activation="relu", use_batch_norm=False)

        self.side_branch1 = SideBranch(in_channels=64, out_channels=32)
        self.side_branch2 = SideBranch(in_channels=256, out_channels=128)

        self.fusion_x1 = ConvBlock(
            32 + 128,
            128,
            kernel_size=3,
            padding=1,
            activation="linear",
            use_batch_norm=False,
        )
        self.fusion_bottleneck = ConvBlock(
            512 + 128,
            512,
            kernel_size=3,
            padding=1,
            activation="linear",
            use_batch_norm=False,
        )

        self.block4 = DecoderBlock(in_channels=512, grow_channels=256, out_channels=256, upsample=True)
        self.block5 = DecoderBlock(in_channels=512, grow_channels=128, out_channels=128, upsample=True)
        self.block6 = DecoderBlock(in_channels=256, grow_channels=64, out_channels=64, upsample=True)

        self.final_head = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, padding=1, activation="leakyrelu"),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        original_size = img.shape[-2:]  # (H, W)
        
        x0 = self.conv_in(img)
        x1 = self.encoder.block1(x0)
        x2 = self.encoder.block2(x1)
        x3 = self.encoder.block3(x2)

        f_x0_side = self.side_branch1(x0)
        x1_fused = self.fusion_x1(torch.cat([f_x0_side, x1], dim=1))

        f_x2_side = self.side_branch2(x2)
        bottleneck_in = self.fusion_bottleneck(torch.cat([x3, f_x2_side], dim=1))

        d4 = self.block4(bottleneck_in)
        d4_skip = torch.cat([d4, x2], dim=1)
        d5 = self.block5(d4_skip)

        d5_skip = torch.cat([d5, x1_fused], dim=1)
        d6 = self.block6(d5_skip)

        output = self.final_head(d6)
        
        return output