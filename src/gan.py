import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
    """Discriminator network for GANs using a configurable CNN architecture.

    This module implements a discriminator network that processes input images
    through a series of convolutional layers defined by a configuration list.

    Attributes
    ----------
    cnn : ConfigurableCNN
        The configurable CNN used for feature extraction and classification.            

    Example
    -------
    >>> img_shape = (3, 64, 64)  # Example image shape
    >>> conv_layers_config = [
    ...     {'out_channels': 64, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'activation': 'leakyrelu'},
    ...     {'out_channels': 128, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'activation': 'leakyrelu'},
    ...     {'out_channels': 256, 'kernel_size': 4, 'stride': 2, 'padding': 1, 'activation': 'leakyrelu'},
    ...     {'out_channels': 1, 'kernel_size': 4, 'stride': 1, 'padding': 0, 'activation': 'linear'},
    ... ]
    >>> discriminator = Discriminator(img_shape, conv_layers_config)
    >>> x = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    >>> output = discriminator(x)
    >>> print(output.shape)
    torch.Size([4, 1, 1, 1])

    Notes
    -----
    - The input images must match the specified `img_shape`.
    - The final output shape depends on the convolutional layers configuration.
    """    
    def __init__(self, img_shape: Tuple[int, int, int], conv_layers_config: List[dict]):
        """Initialize the Discriminator network.

        Parameters
        ----------
        img_shape : Tuple[int, int, int]
            The shape of the input images as (channels, height, width).
        conv_layers_config : List[dict]
            A list of dictionaries specifying the configuration of each convolutional layer.

        Raises
        ------
        RuntimeError
            If the ConfigurableCNN construction fails.
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
        """Forward pass through the Discriminator.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor representing a batch of images with shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the discriminator network.

        Raises
        ------
        TypeError
            If the input is not a torch.Tensor.
        ValueError
            If the input tensor does not have 4 dimensions.
        ValueError
            If the input tensor's shape does not match the expected image shape.
        """        
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
    """Encoder based on the DenseNet121 architecture for feature extraction.

    This class encapsulates a DenseNet121 network (optionally pre-trained) and splits
    its layers into sequential blocks to facilitate access to feature maps at 
    different spatial resolutions.

    Attributes
    ----------
    features : nn.Sequential
        Original feature layers from DenseNet121.
    block1 : nn.Sequential
        First dense block and transition layer, reducing spatial resolution.
    block2 : nn.Sequential
        Second dense block and transition layer.
    block3 : nn.Sequential
        Third dense block and transition layer.
    """
    def __init__(self, pretrained: bool = True):
        """Initializes the DenseNet121 encoder and extracts its feature blocks.

        Parameters
        ----------
        pretrained : bool, optional
            If True, loads ImageNet pre-trained weights. Default is True.
        """
        super(DenseNetEncoder, self).__init__()

        densenet = models.densenet121(
            weights=(models.DenseNet121_Weights.DEFAULT if pretrained else None)
        )
        self.features = densenet.features

        self.block1 = nn.Sequential(self.features.denseblock1, self.features.transition1)
        self.block2 = nn.Sequential(self.features.denseblock2, self.features.transition2)
        self.block3 = nn.Sequential(self.features.denseblock3, self.features.transition3)

class SideBranch(nn.Module):
    """Lateral branch for multi-scale feature fusion.

    Performs downsampling through Average Pooling followed by a 1x1 projection
    convolution to adjust the channel dimension. This is used to inject
    high-resolution information into deeper stages of the network.

    Attributes
    ----------
    proj : ConvBlock
        Convolutional block that performs the linear projection 
        (without Batch Normalization).
    """
    def __init__(self, in_channels: int, out_channels: int):
        """Initializes the lateral branch used for multi-scale feature fusion.

        Parameters
        ----------
        in_channels : int
            Number of channels of the input feature map.
        out_channels : int
            Number of output channels after the 1x1 projection.
        """
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
        """Forward pass through the SideBranch network.

        The input tensor is first downsampled by a factor of 2 using average
        pooling and then passed through a projection layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Projected feature map tensor after downsampling and 1x1 convolution.
        """        
        x_ds = F.avg_pool2d(x, 2)
        return self.proj(x_ds)


class DecoderBlock(nn.Module):
    """Decoder block implementing a dense structure with optional upsampling.

    This block enriches feature maps through a local dense connection, concatenates
    the input with newly generated features, reduces dimensionality, and optionally
    upsamples the spatial resolution.

    Attributes
    ----------
    dense : nn.Sequential
        Dense sub-block (Conv 1x1 -> Conv 3x3).
    up_trans : ConvTransposeBlock
        Transition block that reduces channels via a 1x1 projection.
    """
    def __init__(self, in_channels: int, grow_channels: int, out_channels: int, upsample: bool = True):
        """Initializes the decoder block composed of a dense sub-block, 
        a transition projection block, and an optional upsampling step.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        grow_channels : int
            Growth rate defining how many new channels the dense sub-block generates.
        out_channels : int
            Number of output channels after the projection block.
        upsample : bool, optional
            If True, upsamples the output feature map by a factor of 2 
            using nearest-neighbor interpolation. Default is True.
        """
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

    def _make_dense(self, in_c: int, grow_c: int) -> nn.Sequential:
        """Builds the dense sub-block formed by a 1x1 bottleneck convolution followed by a 3x3 convolution.

        Parameters
        ----------
        in_c : int
            Number of input channels.
        grow_c : int
            Growth rate controlling the number of channels in the 3x3 convolution.

        Returns
        -------
        nn.Sequential
            Dense convolutional block producing `grow_c` new feature channels.
        """
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
        """Forward pass through the DecoderBlock.

        This method processes the input tensor by first applying a dense layer,
        concatenating its output with the original input tensor, and then
        upsampling the result using a transposed convolution. An optional
        additional upsampling step via nearest-neighbor interpolation can be
        applied. The input feature map tensor, expected to have a shape of
        (B, C, H, W), where B is the batch size, C is the number of
        channels, and H, W are the height and width.
        The output upsampled feature map tensor. The spatial dimensions (H, W)
        are increased, and the number of channels is modified by the layers
        within the block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output upsampled feature map tensor with increased spatial dimensions (H, W).
        """        
        dense_feat = self.dense(x)

        x_dense = torch.cat([x, dense_feat], dim=1)

        out = self.up_trans(x_dense)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        return out


class FDGANGenerator(nn.Module):
    """Generator module for the FD-GAN (Fusion-Discriminator GAN) architecture.

    Implements a densely connected U-Net-like structure with lateral side branches
    for multi-scale feature fusion. Designed for image-to-image translation tasks,
    such as haze removal (dehazing).

    Attributes
    ----------
    encoder : DenseNetEncoder
        Pre-trained backbone used for feature extraction.
    conv_in : ConvBlock
        Initial input convolution preserving spatial resolution.
    side_branch1, side_branch2 : SideBranch
        Lateral branches for processing and fusing low-level features.
    fusion_x1, fusion_bottleneck : ConvBlock
        Fusion blocks combining side-branch features with the main encoder stream.
    block4, block5, block6 : DecoderBlock
        Decoder stages for progressively recovering spatial resolution.
    final_head : nn.Sequential
        Final projection layers mapping features to the RGB image space.

    Example
    -------
    >>> generator = FDGANGenerator(output_same_size=True)
    >>> x = torch.randn(4, 3, 256, 256)  # Batch of 4 images
    >>> output = generator(x)
    >>> print(output.shape)
    torch.Size([4, 3, 256, 256])

    References
    ----------
    Dong, Y., Liu, Y., Zhang, H., Chen, S., & Qiao, Y. (2020).
        *FD-GAN: Generative adversarial networks with fusion-discriminator for
        single image dehazing*. AAAI Conference on Artificial Intelligence,
        34(07), 10729-10736.
    """
    def __init__(self, output_same_size=True):
        """Initializes the FD-GAN generator composed of a DenseNet-based encoder,
        lateral side branches, multi-scale fusion modules, decoder stages, 
        and a final reconstruction head.

        Parameters
        ----------
        output_same_size : bool, optional
            Reserved flag indicating whether the output should match the input
            spatial resolution. The current implementation always preserves size.
            Default is True.
        """
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
        """Forward pass through the FD-GAN Generator.
        
        This method implements the U-Net like architecture of the FD-GAN generator,
        which includes an encoder, a decoder, and specialized side-branch and
        fusion modules to combine features from different scales.
        
        The process is as follows:

        1. The input image is passed through an initial convolution.
        2. The result is processed by three encoder blocks to extract features.
        3. Side branches process features from early encoder stages (`x0`, `x2`).
        4. Fusion modules combine these side-branch features with deeper features
            to create inputs for the decoder and a fused skip connection.
        5. The decoder, consisting of three blocks, reconstructs the image, using
            skip connections from the encoder and the fused feature maps.
        6. A final head layer produces the output image.

        Parameters
        ----------
        img : torch.Tensor
            Input normalized image in the range [-1, 1].
            Expected shape: (B, 3, H, W), where H and W are multiples of 32.

        Returns
        -------
        torch.Tensor
            Generated image in the range [-1, 1] with the same spatial resolution
            as the input.
        """
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