from typing import Sequence, Union, Type, Optional, Any, Dict
import torch

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-layer perceptron built from nn.Linear and activation layers.

    Args:
        input_dim (int): Size of the input feature dimension.
        hidden_dims (int | Sequence[int]): Either a single hidden layer size or a sequence
            of hidden layer sizes. If an int is provided, it is treated as a single hidden layer.
        output_dim (int): Size of the output feature dimension.
        activation (Type[nn.Module] | Sequence[Type[nn.Module]], optional): A single activation
            class (e.g. nn.ReLU) to use for all hidden layers, or a sequence of activation
            classes (one per hidden layer). Default: nn.ReLU.

    Notes:
        - No activation is applied after the final (output) linear layer.
        - Activation arguments should be activation classes (callables that return nn.Module
          instances), not instantiated modules.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, Sequence[int]],
        output_dim: int,
        activation: Union[Type[nn.Module], Sequence[Type[nn.Module]]] = nn.ReLU,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        dims = [input_dim] + list(hidden_dims) + [output_dim]

        if not isinstance(activation, (list, tuple)):
            activation = [activation] * (len(dims) - 2)

        assert len(activation) == len(dims) - 2, (
            f"{len(dims) - 2} activation functions expected, "
            f"but {len(activation)} were received"
        )

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(activation[i]())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (..., output_dim)
        """
        return self.layers(x)

class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        """
        Args:
            dim (int): Input and output dimension.
            hidden_dim (int, optional): Intermediate dimension. 
                If not provided, defaults to 2/3 * 4 * dim, following Llama.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # Calculate intermediate dimension
        # Llama uses a multiplier of 2/3 on the "standard" 4*dim
        if hidden_dim is None:
            hidden_dim = int((input_dim* 4) * (2 / 3))
            
        # Ensure hidden_dim is divisible by a good number (e.g., 256) for efficiency
        # This is a common optimization in large models
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Projects to the intermediate dimension (for the gate)
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Projects to the intermediate dimension (for the content)
        self.w_up = nn.Linear(input_dim, hidden_dim, bias=True)

        # Projects back down to the original dimension
        self.w_down = nn.Linear(hidden_dim, input_dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, D)
        """
        # Calculate the gate: SiLU(x @ W_gate)
        # F.silu is the functional form of SiLU
        gate = F.silu(self.w_gate(x))
        
        # Calculate the content: x @ W_up
        up = self.w_up(x)
        
        # Element-wise multiplication (the "gating")
        gated_output = gate * up
        
        # Project down and apply dropout
        output = self.dropout(self.w_down(gated_output))
        
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0
    ):
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

NETS: Dict[str, Any] = {
    'fnn': FeedForwardNetwork,
    'swiglu': SwiGLUFeedForward,
}