'''
References
  [1]: https://arxiv.org/abs/1606.05328
  [2]: https://arxiv.org/abs/1601.06759
  [3]: http://www.scottreed.info/files/iclr2017.pdf
  [4]: https://arxiv.org/abs/1701.05517
'''

import torch
from torch import distributions
from torch import nn

from nets import base


class GatedPixelCNNLayer(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False):
        
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size cannot be even"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = pg_nn.GatedActivation()
        self._kernel_size = kernel_size
        self._padding = (kernel_size - 1) // 2  # (kernel_size - stride) / 2
        self._mask_center = mask_center

        # Vertical stack convolutions.
        self._vstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            kernel_size=(1, self._kernel_size),
            padding=(0, self._padding),
        )
        
        self._vstack_Nx1 = nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(self._kernel_size // 2 + 1, 1),
            padding=(self._padding + 1, 0),
        )
        self._vstack_1x1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2 * out_channels, kernel_size=1
        )

        self._link = nn.Conv2d(
            in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=1
        )

        # Horizontal stack convolutions.
        self._hstack_1xN = nn.Conv2d(
            in_channels=self._in_channels,
            out_channels=2 * self._out_channels,
            kernel_size=(1, self._kernel_size // 2 + 1),
            padding=(0, self._padding + int(self._mask_center)),
        )
        self._hstack_residual = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )
        self._hstack_skip = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, vstack_input, hstack_input):
       
        _, _, h, w = vstack_input.shape  

        # Compute vertical stack.
        vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
        link = self._link(vstack)
        vstack += self._vstack_1x1(vstack_input)
        vstack = self._activation(vstack)

        # Compute horizontal stack.
        hstack = link + self._hstack_1xN(hstack_input)[:, :, :, :w]
        hstack = self._activation(hstack)
        skip = self._hstack_skip(hstack)
        hstack = self._hstack_residual(hstack)
        
        if not self._mask_center:
            hstack += hstack_input

        return vstack, hstack, skip


class GatedPixelCNN(base.AutoregressiveModel):
    """The Gated PixelCNN model."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_gated=10,
        gated_channels=128,
        head_channels=32,
        sample_fn=None,
    ):
        
        super().__init__(sample_fn)
        self._input = GatedPixelCNNLayer(
            in_channels=in_channels,
            out_channels=gated_channels,
            kernel_size=7,
            mask_center=True,
        )
        self._gated_layers = nn.ModuleList(
            [
                GatedPixelCNNLayer(
                    in_channels=gated_channels,
                    out_channels=gated_channels,
                    kernel_size=3,
                    mask_center=False,
                )
                for _ in range(n_gated)
            ]
        )
        self._head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=gated_channels, out_channels=head_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        vstack, hstack, skip_connections = self._input(x, x)
        for gated_layer in self._gated_layers:
            vstack, hstack, skip = gated_layer(vstack, hstack)
            skip_connections += skip
        return self._head(skip_connections)
