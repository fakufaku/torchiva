from typing import List, Optional
import torch

from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply, solve_loaded, solve_loaded_general
from .base import DRBSSBase

def freq_wise_bmm(a, b):
    """
    Parameters
    ----------
    a: (..., n_freq, n_channels_1, n_channels_2)
        op 1
    b: (..., n_channels_2, n_freq, n_frames)
        op 2
    Returns
    -------
    The batch matrix multiplication along the frequency axis,
    then flips the axis back
    Tensor of shape (..., channels_1, n_freq, n_frames)
    """
    return (a @ b.transpose(-3, -2)).transpose(-3, -2)


def batch_abH(a, b):
    """
    Parameters
    ----------
    a: (..., n_channels_1, n_freq, n_frames)
        op 1
    b: (..., n_channels_2, n_freq, n_frames)
        op 2
    Returns
    -------
    Tensor of shape (..., n_freq, n_channels_1, n_channels_2)
    """
    return a.transpose(-3, -2) @ hermite(b.transpose(-3, -2))


def orthogonal_constraint(W_top, Cx, load=1e-4):
    n_src, n_chan = W_top.shape[-2:]

    if n_src == n_chan:
        return W_top
    elif n_chan < n_src:
        raise ValueError(
            "OverIVA requires the number of sources to be "
            "less or equal than that of microphones"
        )

    # create a new demixing matrix
    W = W_top.new_zeros(Cx.shape)
    W[..., :n_src, :] = W_top
    W[..., n_src:, n_src:] = -torch.eye(n_chan - n_src)

    # compute the missing part
    tmp = W_top @ Cx
    W[..., n_src:, :n_src] = hermite(
        solve_loaded_general(tmp[..., :n_src], tmp[..., n_src:], load=load)
    )

    return W


def projection_back_from_demixing_matrix(
    Y: torch.Tensor, W: torch.Tensor, ref_mic: Optional[int] = 0, load: Optional[float] = 1e-4
) -> torch.Tensor:
    """
    Parameters
    ----------
    Y: torch.Tensor (..., n_channels, n_frequencies, n_frames)
        The demixed signals
    W: torch.Tensor (..., n_frequencies, n_channels, n_channels)
        The demixing matrix
    ref_mic: int, optional
        The reference channel
    eps: float, optional
        A diagonal loading factor for the solve method
    """
    # projection back (not efficient yet)
    batch_shape = Y.shape[:-3]
    n_src, n_freq, n_frames = Y.shape[-3:]
    n_chan = W.shape[-1]


    if n_src == n_chan:
        eye = torch.eye(n_chan, n_chan).type_as(W)
        e1 = eye[..., :, [ref_mic]]
        a = solve_loaded(W.transpose(-2, -1), e1)  # (..., n_freq, n_chan, 1)
        a = a.transpose(-3, -2)

    else:
        A = W[..., :n_src, :n_src]
        B = W[..., :n_src, n_src:]
        C = W[..., n_src:, :n_src]

        if ref_mic < n_src:
            eye = torch.eye(n_src, n_src).type_as(W)
            e1 = eye[:, [ref_mic]]
        else:
            e1 = C[..., [ref_mic - n_src], :].transpose(-2, -1)

        WW = A + B @ C
        a = solve_loaded(WW.transpose(-2, -1), e1)
        a = a.transpose(-3, -2)

    Y = Y * a

    return Y, a




class OverIVA_IP(DRBSSBase):
    """
    Over-determined independent vector analysis (IVA) with iterative projection (IP) update [5]_.


    Parameters
    ----------
    n_iter: int, optional
        The number of iterations. (default: ``10``)
    n_src: int, optional
        The number of sources to be separated.
        When ``n_src < n_chan``, a computationally cheaper variant (OverIVA) [6]_ is used.
        If set to ``None``,  ``n_src`` is set to ``n_chan`` (default: ``None``)
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
    forward(X, n_iter=None, n_src=None, model=None, proj_back_mic=None, eps=None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain, 
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated signal in STFT-domain


    References
    ---------
    .. [5] N. Ono,
        "Stable and fast update rules for independent vector analysis based on auxiliary function technique",
        WASSPA, 2011.
    
    .. [6] R. Scheibler, and N Ono,
        "Independent vector analysis with more microphones than sources",
        WASSPA, 2019, https://arxiv.org/pdf/1905.07880.pdf.
    """

    def __init__(
        self,
        n_iter: Optional[int] = 10,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        eps: Optional[float] = None,
    ):

        super().__init__(
            n_iter,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:

        n_iter, n_src, model, proj_back_mic, eps = self._set_params(
            n_iter=n_iter,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

        # remove DC part
        #X = X[..., 1:, :]

        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        # for now, only supports determined case
        assert callable(model)

        W_top = X.new_zeros(batch_shape + (n_freq, n_src, n_chan))
        # minus sign so that the parametrization is correct for overiva
        W_top[:] = torch.eye(n_src, n_chan)

        # initial estimate
        Y = X[..., :n_src, :, :]  # sign to be consistant with W

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = torch.einsum("...cfn,...dfn->...fcd", X, X.conj()) / n_frames

        # apply the orthogonal constraint
        W = orthogonal_constraint(W_top, Cx, load=1e-4)

        for epoch in range(n_iter):

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_frequencies, n_frames)
            weights = model(Y)

            g = torch.clamp(torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps)
            g_sqrt = torch.sqrt(g)
            Y = divide(Y, g_sqrt, eps=eps)
            W_top[:] = divide(W_top, g_sqrt.transpose(-3, -2), eps=eps)
            weights = weights * g

            # we normalize the sources to have source to have unit variance prior to
            # computing the model

            for k in range(n_src):

                W = orthogonal_constraint(W_top, Cx, load=1e-4)

                # compute the weighted spatial covariance matrix
                V = batch_abH(X * weights[..., [k], :, :], X)

                # solve for the new demixing vector
                WV = W @ V

                # the new filter, unscaled
                new_w = torch.conj(
                    #torch.linalg.solve(
                    #    WV + 1e-4 * torch.eye(n_chan, dtype=W.dtype, device=W.device),
                    #    torch.eye(n_chan, dtype=W.dtype, device=W.device)[:, k],
                    #)
                    solve_loaded_general(WV, torch.eye(n_chan, dtype=W.dtype, device=W.device)[:, [k]])[...,0]
                )
                new_w_org = torch.conj(
                    torch.linalg.solve(
                        WV + 1e-4 * torch.eye(n_chan, dtype=W.dtype, device=W.device),
                        torch.eye(n_chan, dtype=W.dtype, device=W.device)[:, k],
                    )
                )

                # resolve scale
                scale = torch.abs(new_w[..., None, :] @ V @ hermite(new_w[..., None, :]))
                new_w = new_w[..., None, :] / torch.sqrt(torch.clamp(scale, min=1e-5))

                # re-build the demixing matrix
                W_top = torch.cat([W[..., :k, :], new_w, W[..., k + 1 : n_src, :]], dim=-2)

                # apply the orthogonal constraint
                #W = orthogonal_constraint(W_top, Cx, load=1e-4)

            

            # demix
            Y = freq_wise_bmm(W_top, X)

            if proj_back_mic is not None:
                Y, a = projection_back_from_demixing_matrix(
                    Y, W, ref_mic=proj_back_mic, load=1e-4
                )

        # add back DC offset
        #pad_shape = Y.shape[:-2] + (1,) + Y.shape[-1:]
        #Y = torch.cat((Y.new_zeros(pad_shape), Y), dim=-2)

        return Y