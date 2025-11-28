# FD-GAN 
## FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing
### Paper Link: [FD-GAN](https://arxiv.org/abs/2001.06968)
### Authors: Yu Dong, Yihao Liu

### Description:
This repository contains the implementation of the FD-GAN model proposed in the paper "FD-GAN: Generative Adversarial Networks with Fusion-discriminator for Single Image Dehazing". The model utilizes a fusion-discriminator to enhance the quality of dehazed images. 

While inspired by the original code, this repository represents a complete refactoring aimed at:

- Modularity: Decomposed architecture into reusable blocks (e.g., ConfigurableCNN).
- Readability: Added type hinting, docstrings, and cleaner logic flow.
- Innovation: Integrated a Score-Matching Regularization term to further guide the generator.

The loss function for the FD-GAN generator proposed in the original paper is given by a combination of the following components:
- Pixel-wise loss
- SSIM loss
- Perceptual loss
- Adversarial loss

For more details, please refer to the original paper linked above.

To improve generation quality and training stability, we introduce a regularization term based on Denoising Score Matching (DSM). We train a separate Score Network ($s_\phi$) to estimate the gradient of the data distribution ( $\nabla_x \log p_{data}(x)$ ). This network acts as an additional critic that is not adversarial, but rather guides the generator towards high-density regions of the natural image manifold. The modified objective function is:

$$L_{total} = L_{FDGAN} + \lambda_{reg} \cdot L_{reg}$$

Where:
- $L$ is the total generator loss.
- $L_{FDGAN}$ is the original generator loss as defined in the paper.
- $L_{reg}$ is the regularization term.
- $\lambda_{reg}$ is a hyperparameter that controls the weight of the regularization term.

The regularization term is defined as the squared norm of the estimated score on generated samples:

$$L_{reg} = \mathbb{E}_{z} \left[ \| s_\phi(G(z)) \|_2^2 \right]$$

By minimizing the norm of the score (implemented as an MSE between the predicted noise/score and zero), we force the generator to produce samples that reside on the "peaks" of the real data distribution as estimated by the score network. This encourages the generator to create more realistic images that align better with the underlying data manifold, thereby improving the overall quality of dehazed images.
