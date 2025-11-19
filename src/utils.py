import torch.nn as nn

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'identity': nn.Identity
}

POOLING_MAP = {
    'max': nn.MaxPool2d,
    'avg': nn.AvgPool2d
}