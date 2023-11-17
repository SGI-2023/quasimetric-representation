from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor, CNN

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding
from ....data.d4rl.type_of_mazes import chosen_maze


class EnvironmentEncoder(nn.Module):
    r"""
    (*, *input_shape)                      Input
           |
     [input_encoding]                      e.g., AtariTorso network to map image input to a flat vector
           |
        (*, d)                             Encoded 1-D input
           |
    [CNN specified by arch]                ENCODER
           |
        (*, latent_size)                   1-D Latent
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (32, 64)
        latent_size: int = 15
        input_shape: int = (2)

        def make(self) -> 'EnvironmentEncoder':
            return EnvironmentEncoder(
                arch=self.arch,
                latent_size=self.latent_size,
                input_shape = self.input_shape
            )

    input_shape: torch.Size
    input_encoding: InputEncoding
    encoder: CNN
    latent_size: int

    def __init__(self, *, arch: Tuple[int, ...], input_shape: int, latent_size: int, **kwargs):
        super().__init__(**kwargs)
        

        self.encoder = CNN(
            input_channels=input_shape,
            output_size=latent_size,
            hidden_channels=arch
        )
        self.latent_size = latent_size
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> LatentTensor:
        one_hot_input = torch.nn.functional.one_hot(x.long()).permute(0, 3, 1, 2).float()
        return self.encoder(one_hot_input)

    # for type hint
    def __call__(self, x: torch.Tensor) -> LatentTensor:
        return super().__call__(x)

    def extra_repr(self) -> str:
        return f"input_shape={self.input_shape}, latent_size={self.latent_size}"
