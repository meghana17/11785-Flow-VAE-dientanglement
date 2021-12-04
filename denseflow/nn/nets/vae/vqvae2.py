'''
References
    [1]: https://arxiv.org/pdf/1906.00446.pdf
    [2]: https://arxiv.org/pdf/1711.00937.pdf
'''

import torch
from torch import nn
from torch.nn import functional as F

from nets import base
import vaes


class VectorQuantizedVAE2(base.GenerativeModel):
    """The VQ-VAE-2 model with a latent hierarchy of depth 2."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_channels=128,
        n_residual_blocks=2,
        residual_channels=32,
        n_embeddings=128,
        embedding_dim=16,
    ):
        
        super().__init__()

        self._encoder_b = vaes.Encoder(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=2,
        )
        self._encoder_t = vaes.Encoder(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=2,
        )
        self._quantizer_t = vaes.Quantizer(
            in_channels=hidden_channels,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
        )
        self._quantizer_b = vaes.Quantizer(
            in_channels=hidden_channels,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
        )
        self._decoder_t = vaes.Decoder(
            in_channels=embedding_dim,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=2,
        )
        self._conv = nn.Conv2d(
            in_channels=hidden_channels, out_channels=embedding_dim, kernel_size=1
        )
        self._decoder_b = vaes.Decoder(
            in_channels=2 * embedding_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_residual_blocks=n_residual_blocks,
            residual_channels=residual_channels,
            stride=2,
        )

    def forward(self, x):
        
        encoded_b = self._encoder_b(x)
        encoded_t = self._encoder_t(encoded_b)

        quantized_t, vq_loss_t = self._quantizer_t(encoded_t)
        quantized_b, vq_loss_b = self._quantizer_b(encoded_b)

        decoded_t = self._decoder_t(quantized_t)
        xhat = self._decoder_b(torch.cat((self._conv(decoded_t), quantized_b), dim=1))
        return xhat, 0.5 * (vq_loss_b + vq_loss_t) + F.mse_loss(decoded_t, encoded_b)

    def sample(self, n_samples):
        raise NotImplementedError("VQ-VAE-2 does not support sampling.")
