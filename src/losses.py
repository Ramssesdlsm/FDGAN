import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import losses
from torchvision import models


class PerceptualLoss(nn.Module):
    """Computes a perceptual similarity metric using early-layer features of VGG16.

    This loss extracts features from the `relu1_2` layer of a pretrained VGG16
    network and applies an L1 distance between the feature activations of two
    images. 

    Notes
    -----
    - The VGG16 backbone is loaded with ImageNet weights and frozen.
    - Only the first four layers of `vgg.features` are used, corresponding to
      the `relu1_2` activation described in the original FD-GAN paper.
    - Input images are expected to be normalized to the appropriate domain
      before calling this loss (e.g., `[0, 1]` range if matching standard VGG preprocessing).

    References
    ----------
    Dong, Y., Liu, Y., Zhang, H., Chen, S., & Qiao, Y. (2020).
        *FD-GAN: Generative adversarial networks with fusion-discriminator for
        single image dehazing*. AAAI Conference on Artificial Intelligence,
        34(07), 10729-10736.

    Examples
    --------
    >>> loss_fn = PerceptualLoss()
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = torch.rand(1, 3, 224, 224)
    >>> loss = loss_fn(x, y)
    >>> loss.backward()
    """    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:4]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the perceptual L1 difference between the VGG16 features of two inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, H, W)``.
        y : torch.Tensor
            Input tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the perceptual loss value.
        """        
        return F.l1_loss(self.feature_extractor(x), self.feature_extractor(y))


class FDGANLoss(nn.Module):
    """Combined loss function used in FD-GAN for single-image dehazing.

    This class implements the generator loss described in the FD-GAN paper,
    which combines multiple components:

    - L1 loss
    - SSIM loss (via Kornia)
    - Perceptual VGG-based loss
    - Adversarial BCE loss

    The final generator loss is computed as::

        L_total = 2 * L1 + 1 * L_ssim + 2 * L_percep + 0.1 * L_adv

    Notes
    -----
    - Inputs for SSIM and perceptual loss must be in the range ``[0, 1]``.
    - Inputs are assumed to be in ``[-1, 1]`` and internally rescaled.
    - The perceptual loss uses frozen VGG16 early-layer features.

    References
    ----------
    Dong, Y., Liu, Y., Zhang, H., Chen, S., & Qiao, Y. (2020).
        *FD-GAN: Generative adversarial networks with fusion-discriminator for
        single image dehazing*. AAAI Conference on Artificial Intelligence,
        34(07), 10729-10736.

    Examples
    --------
    >>> criterion = FDGANLoss()
    >>> fake = torch.rand(1, 3, 256, 256) * 2 - 1
    >>> real = torch.rand(1, 3, 256, 256) * 2 - 1
    >>> pred = torch.randn(1, 1)
    >>> loss, components = criterion(fake, real, pred)
    >>> loss.backward()
    """
    def __init__(self):
        super(FDGANLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = losses.SSIMLoss(window_size=11, reduction="mean")
        self.perceptual = PerceptualLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.perceptual.to(torch.device("cuda"))

    def forward(self, fake, real, disc_pred):
        """Compute the combined FD-GAN generator loss.

        Parameters
        ----------
        fake : torch.Tensor
            Generated image tensor in the range ``[-1, 1]``.
        real : torch.Tensor
            Ground-truth image tensor in the range ``[-1, 1]``.
        disc_pred : torch.Tensor
            Discriminator predictions on the generated images.

        Returns
        -------
        tuple
            A tuple ``(total_loss, components)`` where:

            - **total_loss** : torch.Tensor  
              Scalar tensor with the final weighted sum.

            - **components** : dict  
              Dictionary containing each individual loss term::

                  {
                      "l1":    L1 loss,
                      "ssim":  SSIM loss,
                      "percep": perceptual loss,
                      "adv":   adversarial BCE loss
                  }
        """
        lambda_l1, lambda_ssim, lambda_percep, lambda_adv = 2.0, 1.0, 2.0, 0.1

        loss_l1 = self.l1(fake, real)

        # Denormalize to [0, 1] for SSIM and Perceptual Loss
        fake_01 = (fake + 1) / 2
        real_01 = (real + 1) / 2
        loss_ssim = self.ssim(fake_01, real_01)

        loss_percep = self.perceptual(fake_01, real_01)

        # Adversarial BCE Loss
        loss_adv = self.bce(disc_pred, torch.ones_like(disc_pred))

        total_loss = (
            (lambda_l1 * loss_l1)
            + (lambda_ssim * loss_ssim)
            + (lambda_percep * loss_percep)
            + (lambda_adv * loss_adv)
        )

        return total_loss, {
            "l1": loss_l1,
            "ssim": loss_ssim,
            "percep": loss_percep,
            "adv": loss_adv,
        }


def denoising_score_matching_loss(score_net, clean_samples, sigma=0.1):
    """Computes denoising score matching (DSM) loss.

    This loss perturbs samples with Gaussian noise and trains the network to
    predict the added noise. This approximates score matching without requiring
    explicit computation of the data score.

    Parameters
    ----------
    score_net : callable
        Neural network that maps noisy samples to predicted noise.
    clean_samples : torch.Tensor
        Clean input samples of any shape.
    sigma : float, optional
        Standard deviation of the Gaussian perturbation. Default is ``0.1``.

    Returns
    -------
    torch.Tensor
        Scalar loss value representing the MSE between predicted and actual noise.

    Notes
    -----
    This formulation follows standard denoising score matching used in
    diffusion models and score-based generative modeling.

    Examples
    --------
    >>> net = lambda x: torch.zeros_like(x)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> loss = denoising_score_matching_loss(net, x)
    """
    noise = torch.randn_like(clean_samples)
    perturbed_samples = clean_samples + noise * sigma

    predicted_noise = score_net(perturbed_samples)

    loss = F.mse_loss(predicted_noise, noise)

    return loss
