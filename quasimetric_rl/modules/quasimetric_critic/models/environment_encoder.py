from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

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
    [MLP specified by arch]                ENCODER
           |
        (*, latent_size)                   1-D Latent
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        latent_size: int = 128
        input_shape: int = len(chosen_maze)

        def make(self) -> 'EnvironmentEncoder':
            return EnvironmentEncoder(
                arch=self.arch,
                latent_size=self.latent_size,
                input_shape = self.input_shape
            )

    input_shape: torch.Size
    input_encoding: InputEncoding
    encoder: MLP
    latent_size: int

    def __init__(self, *, arch: Tuple[int, ...], input_shape: int, latent_size: int, **kwargs):
        super().__init__(**kwargs)
        

        self.encoder = MLP(input_shape, latent_size, hidden_sizes=arch)
        self.latent_size = latent_size
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor) -> LatentTensor:
        return self.encoder(x)

    # for type hint
    def __call__(self, x: torch.Tensor) -> LatentTensor:
        return super().__call__(x)

    def extra_repr(self) -> str:
        return f"input_shape={self.input_shape}, latent_size={self.latent_size}"
