from typing import Sequence, Union, Type
import torch

import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron built from nn.Linear and activation layers.

    This module implements a feedforward neural network of the form:

    .. math::

        f(x) = W_n \sigma( W_{n-1} \sigma(\dots \sigma(W_1 x + b_1)\dots ) + b_{n-1}) + b_n

    Attributes
    ----------
    layers : nn.Sequential
        The sequential container holding all linear and activation layers.

    Example
    -------
    >>> mlp = MLP(input_dim=10, hidden_dims=[20, 30], output_dim=5, activation=nn.ReLU)
    >>> x = torch.randn(4, 10)  # Batch of 4 samples
    >>> output = mlp(x)
    >>> print(output.shape)
    torch.Size([4, 5])
        
    Notes
    -----
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
        """Initialize the Multilayer Perceptron (MLP).

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        hidden_dims : int or Sequence[int]
            Size(s) of the hidden layer(s).
        output_dim : int
            Dimension of the output features.
        activation : Type[nn.Module] or Sequence[Type[nn.Module]], optional
            Activation function(s) to use. Defaults to nn.ReLU.
        
        Notes
        -----
        The number of hidden layers equals ``len(hidden_dims)``.
        If ``activation`` is a single class, the same activation is applied to all hidden layers.

        """
        super(MLP, self).__init__()

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
        """Forward pass through the MLP.
        
        The MLP is applied independently to the last dimension, so additional batch
        dimensions are preserved.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim).
        """
        return self.layers(x)