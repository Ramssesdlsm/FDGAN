import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    """A simple convolutional neural network for score estimation.
    
    This module implements a small encoder-decoder architecture using convolutional
    and transposed convolutional layers with ELU activations to estimate the score 
    function :math:`\nabla_x \log p(x)`.

    Attributes
    ----------
    main : nn.Sequential
        The sequential model containing the convolutional and transposed
        convolutional layers.
    """    
    def __init__(self, channels: int = 3):
        """Initialize the ScoreNet model.

        Parameters
        ----------
        channels : int, optional
            Number of input and output channels. For RGB images is typically 3. 
            Default is 3.
        """        
        super(ScoreNet, self).__init__()
        self.main = nn.Sequential(
            # Down
            nn.Conv2d(channels, 64, 3, 1, 1), nn.ELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ELU(), # 128 x H/2 x W/2
            nn.Conv2d(128, 256, 3, 2, 1), nn.ELU(), # 256 x H/4 x W/4
            # Up
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(), # Back to H/2 x W/2
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),  # Back to H x W
            nn.Conv2d(64, channels, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the ScoreNet.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as the input, representing the 
            estimated score for each pixel.
        """        
        return self.main(x)