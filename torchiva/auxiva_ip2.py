# Copyright (c) 2022 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Optional
import torch

from .linalg import divide, eigh, hermite, inv_2x2, mag_sq, eigh_2x2
from .models import LaplaceModel
from .base import DRBSSBase


def spatial_model_update_ip2(
    Xo: torch.Tensor,
    weights: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
):
    """
    Apply the spatial model update via the generalized eigenvalue decomposition.
    This method is specialized for two channels.

    Parameters
    ----------
    Xo: torch.Tensor, shape (..., n_frequencies, n_channels, n_frames)
        The microphone input signal with n_chan == 2
    weights: torch.Tensor, shape (..., n_frequencies, n_channels, n_frames)
        The weights obtained from the source model to compute
        the weighted statistics

    Returns
    -------
    X: torch.Tensor, shape (n_frequencies, n_channels, n_frames)
        The updated source estimates
    """
    assert Xo.shape[-3] == 2, "This method is specialized for two channels processing."

    V = []
    for k in [0, 1]:
        # shape: (n_batch, n_freq, n_chan, n_chan)
        Vloc = torch.einsum(
            "...fn,...cfn,...dfn->...fcd", weights[..., k, :, :], Xo, Xo.conj()
        )
        Vloc = Vloc / Xo.shape[-1]
        # make sure V is hermitian symmetric
        Vloc = 0.5 * (Vloc + hermite(Vloc))
        V.append(Vloc)

    # we use the custom 2x2 eigenvalue decomposition solver. It is probably
    # not always faster than torch, but should be more stable for gradient
    # computation
    eigval, eigvec = eigh_2x2(V[1], V[0], eps=eps)

    # reverse order of eigenvectors
    eigvec = torch.flip(eigvec, dims=(-1,))

    scale_0 = abs(
        torch.conj(eigvec[..., None, :, 0]) @ (V[0] @ eigvec[..., :, None, 0])
    )
    scale_1 = abs(
        torch.conj(eigvec[..., None, :, 1]) @ (V[1] @ eigvec[..., :, None, 1])
    )
    scale = torch.cat((scale_0, scale_1), dim=-1)
    scale = torch.clamp(torch.sqrt(torch.clamp(scale, min=1e-7)), min=eps)
    eigvec = eigvec / scale

    if W is not None:
        W = hermite(eigvec)
        if A is not None:
            A = inv_2x2(W)

    X = torch.einsum("...fcd,...dfn->...cfn", hermite(eigvec), Xo)

    return X, W, A


class AuxIVA_IP2(DRBSSBase):
    """
    Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors [7]_

    Parameters
    ----------
    n_iter: int, optional
        The number of iterations (default: ``10``).
    model: torch.nn.Module, optional
        The model of source distribution.
        If ``None``, spherical Laplace is used (default: ``None``).
    proj_back_mic: int, optional
        The reference mic index to perform projection back.
        If set to ``None``, projection back is not applied (default: ``0``).
    eps: float, optional
        A small constant to make divisions and the like numerically stable (default:``None``).


    Methods
    --------
    forward(X, n_iter=None, model=None, proj_back_mic=None, eps=None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    -------
    X: torch.Tensor, ``shape (..., n_chan, n_freq, n_frames)``
        The separated signal in STFT-domain.

    References
    ----------
    .. [7] N. Ono,
        "Fast stereo independent vector analysis and its implementation on mobile phone",
        IWAENC, 2012.
    """

    def __init__(
        self,
        n_iter: Optional[int] = 10,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        eps: Optional[float] = None,
    ):

        super().__init__(
            n_iter,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

        # the different parts of the demixing matrix
        self.W = None  # target sources
        self.A = None  # mixing matrix

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:

        n_chan, n_freq, n_frames = X.shape[-3:]

        n_iter, model, proj_back_mic, eps = self._set_params(
            n_iter=n_iter,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

        # for now, only supports determined case
        assert callable(model)

        # initialize source model if NMF
        self._reset(model)

        # only supports two channels case in IP2
        assert n_chan == 2

        Xo = X

        if proj_back_mic is not None:
            assert (
                0 <= proj_back_mic < n_chan
            ), "The reference microphone index must be between 0 and # channels - 1."
            W = X.new_zeros(X.shape[:-3] + (n_freq, n_chan, n_chan))
            W[:] = torch.eye(n_chan).type_as(W)
            A = W.clone()
        else:
            W = None
            A = None

        for epoch in range(n_iter):

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_frequencies, n_frames)
            weights = model(X)

            # we normalize the sources to have source to have unit variance prior to
            # computing the model
            g = torch.clamp(torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
            X = divide(X, torch.sqrt(g))
            weights = weights * g

            # Here are the exact/fast updates for two channels using the GEVD
            X, W, A = spatial_model_update_ip2(Xo, weights, W=W, A=A, eps=eps)

        if proj_back_mic is not None and n_iter > 0:
            a = A[..., :, [proj_back_mic], :].moveaxis(-1, -3)
            X = a * X

        return X
