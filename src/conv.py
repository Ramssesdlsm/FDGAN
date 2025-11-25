import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from .utils import ACTIVATION_MAP, POOLING_MAP

class ConvBlock(nn.Module):
    """A modular convolutional block for PyTorch models.

    A configurable 2D convolutional block that composes a Conv2d layer with optional BatchNorm,
    a selectable activation, and an optional pooling layer. The block is assembled into an
    nn.Sequential and applied as-is in forward.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image/tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int
        Size of the convolving kernel (assumed square).
    stride : int, optional (default=1)
        Stride of the convolution.
    padding : Union[int, str, bool], optional (default=False)
        Padding applied to the input before convolution. Behaviors:

        - If the string ``same`` (case-insensitive) is provided, padding is set to ``kernel_size // 2``
            (to approximately preserve spatial dimensions for odd kernel sizes).
        - If a positive integer is provided, that value is used as padding.
        - Any other value (including ``False`` or ``0``) results in zero padding.

    activation : str, optional (default=``'relu'``)
        Name of the activation to apply after convolution (looked up via ``ACTIVATION_MAP``).
    activation_kwargs : Optional[Dict[str, Any]], optional (default=None)
        Optional keyword arguments to instantiate the activation class. If None, the activation is created with its default constructor.
    use_batch_norm : bool, optional (default=True)
        If True, a nn.BatchNorm2d(out_channels) layer is inserted after the convolution.
        When batch normalization is used, the convolution is created with bias=False.
    pooling_type : Optional[str], optional
        Name of the pooling operation to apply (looked up via ``POOLING_MAP``), e.g. ``max``, ``avg``.
        If None, no pooling layer is appended.
    pooling_kernel : Optional[int], optional
        Kernel size for the pooling layer. If not provided, pooling is not added.
    pooling_stride : Optional[int], optional
        Stride for the pooling layer. If None, defaults to pooling_kernel.

    Attributes
    ----------
    block : nn.Sequential
        The assembled sequential block containing the convolution, optional batch norm,
        activation, and optional pooling.

    Raises
    ------
    ValueError
        If pooling_type is provided but not found in POOLING_MAP.

    Notes
    -----
    - The convolution's bias is disabled when batch normalization is enabled to avoid redundant
      affine parameters.
    - Activation and pooling implementations are looked up using ACTIVATION_MAP and POOLING_MAP,
      respectively; those mappings must be defined in the module scope and map lowercase names
      to callable nn.Module classes (not instances).
    - The forward pass simply delegates to self.block(x) and returns the transformed tensor.

    Example
    -------
    >>> # Construct a conv block with same-padding, batch norm and max pooling
    >>> ConvBlock(3, 64, kernel_size=3, padding="same", activation="relu",
    ...           use_batch_norm=True, pooling_type="max", pooling_kernel=2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str, bool] = False,
        activation: str = "relu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        use_batch_norm: bool = False,
        pooling_type: Optional[str] = None,
        pooling_kernel: Optional[int] = None,
        pooling_stride: Optional[int] = None,
    ):
        super(ConvBlock, self).__init__()

        padding_value = 0
        if isinstance(padding, str) and padding.lower() == "same":
            padding_value = kernel_size // 2
        elif isinstance(padding, int) and padding > 0:
            padding_value = padding

        layers = []

        conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding_value,
            bias=not use_batch_norm,
        )
        layers.append(conv_layer)

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        activation_class = ACTIVATION_MAP.get(activation.lower(), nn.ReLU)
        layers.append(activation_class(**(activation_kwargs or {})))

        if pooling_type and pooling_kernel:
            pooling_class = POOLING_MAP.get(pooling_type.lower())
            if not pooling_class:
                raise ValueError(f"Pooling type '{pooling_type}' is not supported.")
            pool_stride = (
                pooling_stride if pooling_stride is not None else pooling_kernel
            )
            layers.append(pooling_class(kernel_size=pooling_kernel, stride=pool_stride))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """A modular transposed-convolution block for PyTorch models.

    A modular transposed-convolution block for PyTorch models that performs a ConvTranspose2d
    followed optionally by BatchNorm2d and a configurable activation.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    out_channels : int
        Number of channels produced by the transposed convolution.
    kernel_size : int
        Size of the convolution kernel.
    stride : int, optional
        Stride of the transposed convolution. Default is 2.
    padding : int, optional
        Padding added to both sides of the input. Default is 1.
    output_padding : int, optional
        Additional size added to one side of the output shape. Default is 0.
    activation : str, optional
        Name of the activation to apply after convolution (looked up via ACTIVATION_MAP).
        Defaults to "relu". The activation class is instantiated with its default constructor.
    activation_kwargs : Optional[Dict[str, Any]], optional (default=None)
        Optional keyword arguments to instantiate the activation class. If None, the activation is created with its default constructor.
    use_batch_norm : bool, optional
        If True, insert nn.BatchNorm2d after the transposed convolution. Default is True.

    Behavior
    --------
    - Constructs an nn.Sequential block containing:
        1. nn.ConvTranspose2d(..., bias=not use_batch_norm)
        2. nn.BatchNorm2d(out_channels) if use_batch_norm is True
        3. The activation layer resolved from ACTIVATION_MAP
    - The ConvTranspose2d layer's bias is disabled when batch normalization is used
      (bias=False) to avoid redundant affine parameters.

    Input / Output shapes
    ---------------------
    Input:  Tensor of shape (N, in_channels, H, W)
    Output: Tensor of shape (N, out_channels, H_out, W_out), where H_out and W_out depend
            on kernel_size, stride, padding, and output_padding.

    Examples
    --------
    >>> block = ConvTransposeBlock(128, 64, kernel_size=4, stride=2, padding=1)
    >>> out = block(x)  # x: (N, 128, H, W) -> out: (N, 64, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        activation: str = "relu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        use_batch_norm: bool = False,
    ):
        super(ConvTransposeBlock, self).__init__()

        layers = []

        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=not use_batch_norm,
            )
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        activation_class = ACTIVATION_MAP.get(activation.lower(), nn.ReLU)
        layers.append(activation_class(**(activation_kwargs or {})))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConfigurableCNN(nn.Module):
    """A customizable convolutional neural network (CNN) built from a sequence of ConvBlock.

    A customizable convolutional neural network (CNN) built from a sequence of ConvBlock
    layers and ConvTransposeBlock layers defined by the user.

    Parameters
    ----------
    layers_config : List[Dict[str, Any]]
        A list of dictionaries, each specifying the parameters for a ConvBlock or ConvTransposeBlock.
        Each dictionary should contain keys corresponding to the ConvBlock's
        constructor parameters (e.g., in_channels, out_channels, kernel_size, etc.).

    Attributes
    ----------
    cnn : nn.Sequential
        The assembled sequential CNN composed of the specified ConvBlock and ConvTransposeBlock layers.

    Example
    -------
    >>> layers_config = [
    ...     {conv:{"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": "same"}},
    ...     {conv:{"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": "same", "pooling_type": "max", "pooling_kernel": 2}},
    ...     {conv_transpose:{"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": "same"}}
    ... ]
    >>> model = ConfigurableCNN(layers_config)
    """

    def __init__(self, layers_config):
        super(ConfigurableCNN, self).__init__()

        layers = []
        for layer_cfg in layers_config:
            if not isinstance(layer_cfg, dict) or len(layer_cfg) != 1:
                raise ValueError("Each layer configuration must be a dictionary with only one key.")
            
            layer_type = next(iter(layer_cfg))
            layer_params = layer_cfg[layer_type]

            if layer_type == "conv":
                layers.append(ConvBlock(**layer_params))
            elif layer_type == "conv_transpose":
                layers.append(ConvTransposeBlock(**layer_params))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.cnn = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.cnn(x)
    
if __name__ == "__main__":
    # Example usage
    layers_config = [
        {"conv": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "padding": "same"}},
        {"conv": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": "same", "pooling_type": "max", "pooling_kernel": 2}},
        {"conv_transpose": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": "same"}}
    ]
    model = ConfigurableCNN(layers_config)
    print(model)