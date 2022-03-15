from typing import Optional

import torch

from .linalg import hankel_view, mag_sq, solve_loaded


def derev(H: torch.Tensor, X: torch.Tensor, X_bar: torch.Tensor):
    return X - torch.einsum("...fcts,...cftn->...sfn", H.conj(), X_bar)


def wpe_default_weights(Y: torch.Tensor, eps: Optional[float] = 1e-5) -> torch.Tensor:
    w = 1.0 / torch.clamp(torch.mean(mag_sq(Y), dim=-3), min=eps)
    w = w / w.sum(dim=-1, keepdim=True)
    return w


def wpe_one_iter(
    Y: torch.Tensor,
    X: torch.Tensor,
    X_bar: torch.Tensor,
    model: Optional[callable] = None,
    eps: Optional[float] = 0.0,
) -> torch.Tensor:
    """
    Parameters
    ----------
    Y: torch.Tensor, (..., n_chan, n_freq, n_frames)
        The current estimate of the dereverberated signal
    X: torch.Tensor, (..., n_chan, n_freq, n_frames)
        Input signal
    X_bar: torch.Tensor, (..., n_chan, n_freq, n_taps, n_frames)
        Delayed version of input signal

    Returns
    -------
    H: torch.Tensor, (..., n_freq, n_chan, n_taps, n_chan)
        The updated dereverberation filter weights
    """
    batch_shape = X_bar.shape[:-4]
    n_chan, n_freq, n_taps, n_frames = X_bar.shape[-4:]
    Lh = n_taps * n_chan

    if model is None:
        model = wpe_default_weights

    weights = model(Y)

    # compute weighted statistics
    acm = torch.einsum("...fn,...cftn,...dfun->...fctdu", weights, X_bar, X_bar.conj())
    xcv = torch.einsum("...fn,...cftn,...sfn->...fcts", weights, X_bar, X.conj())

    # solve the system
    acm = acm.reshape(batch_shape + (n_freq, Lh, Lh))
    xcv = xcv.reshape(batch_shape + (n_freq, Lh, n_chan))

    # H = torch.linalg.solve(acm + eps * torch.eye(Lh).type_as(acm), xcv)
    H = torch.linalg.solve(acm, xcv)

    H = H.reshape(batch_shape + (n_freq, n_chan, n_taps, n_chan))

    return H


def wpe(
    X: torch.Tensor,
    n_iter=10,
    n_delay=3,
    n_taps=5,
    H0: Optional[torch.Tensor] = None,
    model: Optional[callable] = None,
    eps: Optional[float] = 1e-5,
    verbose: Optional[bool] = False,
):

    batch_shape = X.shape[:-3]
    n_chan, n_freq, n_frames = X.shape[-3:]

    # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
    X_pad = torch.nn.functional.pad(X, (n_taps + n_delay, 0))
    X_hankel = hankel_view(X_pad, n_taps + n_delay + 1)
    X_bar = X_hankel[..., : -n_delay - 1, :]  # shape (c, f, t, b)

    # the dereverb weights
    if H0 is None:
        # default init at zero
        H = X.new_zeros(batch_shape + (n_freq, n_chan, n_taps, n_chan))
        Y = X.clone()
    else:
        H = H0.clone()
        Y = derev(H, X, X_bar)

    for epoch in range(n_iter):
        H = wpe_one_iter(Y, X, X_bar, model=model, eps=eps)
        Y = derev(H, X, X_bar)

    return Y, H
