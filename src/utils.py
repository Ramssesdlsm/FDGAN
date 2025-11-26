import os
from typing import Callable, Union

import kornia.filters
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}

POOLING_MAP = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}


def get_lf_hf(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the low- and high-frequency components of a batch of images.

    This function extracts two complementary frequency representations from an
    input batch of images. The low-frequency (LF) component is obtained using a
    Gaussian blur, which removes fine details while preserving coarse structure.
    The high-frequency (HF) component is computed by applying a Laplacian filter
    to the grayscale version of the input and then replicating the result across
    the RGB channels. Because the Laplacian may produce high-magnitude values,
    the HF output is normalized to the range ``[-1, 1]`` using a hyperbolic
    tangent to ensure numerical stability during training.

    Parameters
    ----------
    images : torch.Tensor
        Input batch of images with shape ``(B, C, H, W)`` and RGB channels.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A pair ``(lf, hf)`` where:

        - ``lf`` is the low-frequency component obtained via Gaussian blur.
        - ``hf`` is the high-frequency component derived from the Laplacian and
          normalized to ``[-1, 1]``.

    Notes
    -----
    - The HF component is expanded to three channels so that both LF and HF can
      be concatenated with the original RGB image if needed (e.g., for
      discriminators in GAN architectures).
    - Normalization with ``tanh`` prevents unstable gradients that may arise
      from the Laplacian operator.
    """
    lf = TF.gaussian_blur(images, kernel_size=15, sigma=3.0)

    gray = TF.rgb_to_grayscale(images)
    laplacian = kornia.filters.laplacian(gray, kernel_size=3)
    hf = laplacian.repeat(1, 3, 1, 1)

    hf = torch.tanh(hf)

    return lf, hf


def prepare_discriminator_input(
    img: torch.Tensor, lf: torch.Tensor, hf: torch.Tensor
) -> torch.Tensor:
    """Concatenate an image batch with its low- and high-frequency components.

    This function prepares the input for a discriminator by concatenating, along
    the channel dimension, the original RGB images together with their
    corresponding low-frequency (LF) and high-frequency (HF) representations.
    The resulting tensor has nine channels: three from the original image,
    three from the LF component, and three from the HF component.

    Parameters
    ----------
    img : torch.Tensor
        Batch of RGB images with shape ``(B, 3, H, W)``.
    lf : torch.Tensor
        Batch of low-frequency components with shape ``(B, 3, H, W)``.
    hf : torch.Tensor
        Batch of high-frequency components with shape ``(B, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Concatenated tensor of shape ``(B, 9, H, W)`` corresponding to
        ``[img or lf or hf]`` along the channel dimension.
    """
    return torch.cat([img, lf, hf], dim=1)


def weights_init_normal(m: nn.Module) -> None:
    """Initialize model weights following a normal distribution scheme.

    This function is intended to be passed to ``nn.Module.apply`` to initialize
    convolutional and batch-normalization layers commonly used in GANs.
    Convolutional layers are initialized with a normal distribution centered at
    0, while batch-normalization layers are initialized with weights centered at
    1 and biases at 0.

    Parameters
    ----------
    m : nn.Module
        A PyTorch module. If ``m`` is an instance of ``Conv2d`` or
        ``ConvTranspose2d``, its weights are initialized using
        ``N(0, 0.02)``. If ``m`` is a ``BatchNorm2d`` or ``BatchNorm1d`` layer,
        its weights are initialized using ``N(1, 0.02)`` and its biases are set
        to zero.

    Notes
    -----
    - This initialization scheme follows standard practices in GAN training
      (e.g., DCGAN), improving stability during early optimization.
    - Use as: ``model.apply(weights_init_normal)``.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DehazingDataset(Dataset):
    """A paired-image dataset for single-image dehazing tasks.

    This dataset assumes a directory structure:

    ``root_dir/clear`` — clean (ground-truth) images
    ``root_dir/hazy`` — corresponding hazy/degraded images

    Both subdirectories must contain images with matching filenames.
    The dataset returns pairs ``(hazy, clear)`` suitable for supervised
    training of dehazing models.

    Parameters
    ----------
    root_dir : str or Path-like
        Path to the dataset root directory containing ``clear/`` and ``hazy/``.
    transform : callable, optional
        A function or transform applied to both images. It must accept and
        return a PIL image or a tensor. When provided, the same transform is
        applied to both hazy and clear images.

    Attributes
    ----------
    root_dir : str
        Path to the dataset root.
    clear_dir : str
        Directory containing the clean images.
    hazy_dir : str
        Directory containing the hazy images.
    images : list of str
        Filenames present in ``clear_dir`` (matching files are expected in
        ``hazy_dir``).
    transform : callable or None
        Transform applied to both images in a sample.
    """

    def __init__(self, root_dir: str, transform: Union[Callable, None] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.clear_dir = os.path.join(root_dir, "clear")
        self.hazy_dir = os.path.join(root_dir, "hazy")
        self.images = [
            f for f in os.listdir(self.clear_dir) if f.endswith((".jpg", ".png"))
        ]

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> tuple[Union[torch.Tensor, Image.Image], Union[torch.Tensor, Image.Image]]:
        """Load and return a hazy/clear image pair.

        Parameters
        ----------
        idx : int
            Index of the image pair to load.

        Returns
        -------
        tuple[torch.Tensor or PIL.Image, torch.Tensor or PIL.Image]
            A tuple ``(hazy_img, clear_img)``. If a transform is provided,
            both elements are tensors. Otherwise, they are returned as PIL
            images converted to RGB.
        """
        img_name = self.images[idx]

        clear_path = os.path.join(self.clear_dir, img_name)
        hazy_path = os.path.join(self.hazy_dir, img_name)

        clear_img = Image.open(clear_path).convert("RGB")
        hazy_img = Image.open(hazy_path).convert("RGB")

        if self.transform:
            clear_img = self.transform(clear_img)
            hazy_img = self.transform(hazy_img)

        return hazy_img, clear_img
