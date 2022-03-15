from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

class MultiBLSTMMask(nn.Module):
    """
    This module procures multiple masks for multiple sources.

    C. Boeddeker et al., "CONVOLUTIVE TRANSFER FUNCTION INVARIANT SDR TRAINING
    CRITERIA FOR MULTI-CHANNEL REVERBERANT SPEECH SEPARATION", Proc. ICASSP
    2021.
    """

    def __init__(
        self,
        *args,
        n_src: Optional[int] = 1,
        n_masks: Optional[int] = 3,
        n_input: Optional[int] = 512,
        n_output: Optional[int] = None,
        n_hidden: Optional[int] = 300,
        dropout_p: Optional[float] = 0.5,
        n_layers: Optional[int] = 3,
        norm_time: Optional[bool] = False,
        eps: Optional[float] = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.n_src = n_src
        self.n_masks = n_masks
        self.norm_time = norm_time

        if n_output is None:
            n_output = n_input

        self.blstm = nn.LSTM(
            input_size=n_input,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.ff1 = torch.nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.ff2 = torch.nn.Linear(2 * n_hidden, n_masks * n_src * n_output)

    def forward(self, X):
        """
        Returns
        -------
        masks: torch.Tensor (..., n_src, n_masks, n_freq, n_frames)
            Multiple masks for multiple sources, each the size of the spectorgram
        """
        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        # flatten the batch
        X = X.reshape((-1, n_freq, n_frames))

        # input features
        X = torch.log(1.0 + X.abs())

        # BLSTM
        X = X.permute([2, 0, 1])  # -> (n_frames, n_batch, n_freq)
        X = self.blstm(X)[0]  # -> (n_frames, n_batch, n_hidden * 2)

        # linear 1
        X = F.relu(self.ff1(X))

        # linear 2
        X = torch.sigmoid(self.ff2(X))

        # re-order
        X = X.permute([1, 2, 0])  # -> (n_batch, n_freq, n_frames)

        X = (1 - self.eps) * X + self.eps

        # restore batch size
        X = X.reshape(batch_shape + X.shape[-2:])

        X = X.reshape(X.shape[:-2] + (self.n_src, self.n_masks, -1) + X.shape[-1:])

        if self.norm_time:
            X = X / torch.sum(X, dim=-1, keepdim=True)

        return X


class BLSTMMask(MultiBLSTMMask):
    """
    This module procures one mask per input spectrogram.
    """

    def __init__(
        self,
        *args,
        n_input=512,
        n_output=None,
        n_hidden=300,
        dropout_p=0.5,
        n_layers=3,
        norm_time=False,
        eps=1e-3,
    ):
        super().__init__(
            n_src=1,
            n_masks=1,
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            dropout_p=dropout_p,
            n_layers=n_layers,
            norm_time=norm_time,
            eps=eps,
        )

    def forward(self, x):
        inshape = x.shape
        x = MultiBLSTMMask.forward(self, x)
        # lose the src and mask dimensions
        x = x[..., 0, 0, :, :]

        x = torch.broadcast_to(x, inshape)
        return x

