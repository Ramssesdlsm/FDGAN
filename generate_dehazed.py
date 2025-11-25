import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gan import FDGANGenerator


def pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized PyTorch image tensor into a PIL Image.

    This function expects an image tensor in the range ``[-1, 1]``.
    The tensor is detached from the computation graph, moved to CPU, and
    rescaled back to ``[0, 1]`` before converting to a ``uint8`` NumPy array
    suitable for PIL.

    Only the first image of the batch is converted (i.e., ``tensor[0]``).

    Parameters
    ----------
    tensor : torch.Tensor
        A tensor representing a batch of images with shape ``(B, C, H, W)``.
        Values should be in the range ``[-1, 1]``. The function converts only
        the first sample of the batch.

    Returns
    -------
    Image.Image
        A PIL Image in RGB format created from the first tensor in the batch.
    """
    t = tensor.detach().cpu().clamp(-1, 1)
    t = (t + 1) / 2
    t = t[0]
    arr = (t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dehazed image using a trained FDGAN generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        default="./checkpoints/SSM_Model/gen_epoch_19.pth",
        help="Path to the generator .pth file. Default: ./checkpoints/SSM_Model/gen_epoch_19.pth",
    )
    parser.add_argument(
        "--input", required=True, help="Input hazy image (JPG, PNG, etc.)"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for the dehazed image"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Image size (height width) to process. If not specified, the original size is used.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda or cpu). Default cuda if available.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading generator on device={device}...")
    gen = FDGANGenerator(output_same_size=True).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load the generator state
    state = torch.load(args.checkpoint, map_location=device)

    gen.load_state_dict(state)

    gen.eval()

    # We define the transform here to include resizing and a normalization to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size)
            if args.img_size
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img = Image.open(args.input).convert("RGB")

    inp = transform(img).unsqueeze(0).to(device)

    print("Generating dehazed image...")
    with torch.no_grad():
        # We pass the input through the generator
        out = gen(inp)

    # We create the imagen from the output tensor
    out_img = pil_from_tensor(out)
    out_dir = os.path.dirname(args.output)
    # We create the output directory if it does not exist
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_img.save(args.output)
    print(f"Dehazed image saved to: {args.output}")


if __name__ == "__main__":
    main()
