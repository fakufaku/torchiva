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

from typing import NoReturn, Optional

import torch as pt
import torch.nn as nn

from .linalg import mag, mag_sq
from .parameters import eps_scaling


class Scaling(nn.Module):
    """
    We should implement the scaling step as a pytorch model
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X


def projection_back(Y: pt.Tensor, ref: pt.Tensor) -> NoReturn:
    """
    Solves the scale ambiguity according to Murata et al., 2001.
    This technique uses the steering vector value corresponding
    to the demixing matrix obtained during separation.

    Parameters
    ----------
    Y: torch.Tensor (n_batch, n_channels, n_frequencies, n_frames)
        The STFT data to project back on the reference signal
    ref: torch.Tensor (..., n_frequencies, n_frames)
        The reference signal
    """
    shape = Y.shape[:-3]
    n_chan, n_freq, n_frames = Y.shape[-3:]

    Y = Y.transpose(-3, -2)  # put Y in order (..., n_freq, n_chan, n_frames)

    # find a bunch of non-zero frames
    I_nz = pt.argsort(pt.sum(mag(Y), dim=(-3, -2)), dim=-1)[..., -n_chan:]

    I_nz_Y, _ = pt.broadcast_tensors(I_nz[..., None, None, :], Y[..., :n_chan])
    I_nz_ref, _ = pt.broadcast_tensors(I_nz[..., None, :], ref[..., :n_chan])

    A = pt.gather(Y, dim=-1, index=I_nz_Y).transpose(-2, -1)
    b = pt.gather(ref, dim=-1, index=I_nz_ref)
    b = b[..., None]

    # Now we only need to solve a linear system of size n_chan x n_chan
    # per frequency band
    A_flat = A.reshape((-1, n_chan, n_chan))
    b_flat = b.reshape((-1, n_chan, b.shape[-1]))

    dload = 1e-5 * pt.eye(n_chan, dtype=A_flat.dtype, device=A_flat.device)
    c = pt.linalg.solve(A_flat + dload, b_flat)

    return (Y * c.reshape(shape + (n_freq, n_chan, 1))).transpose(-3, -2)


def minimum_distortion_l2(Y: pt.Tensor, ref: pt.Tensor) -> pt.Tensor:
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Parameters
    ----------
    Y: torch.Tensor (..., n_channels, n_frequencies, n_frames)
        The STFT data to project back on the reference signal
    ref: torch.Tensor (..., n_frequencies, n_frames)
        The reference signal
    """
    num = pt.sum(ref[..., None, :, :].conj() * Y, dim=-1) / Y.shape[-1]
    denom = mag_sq(Y).mean(-1)

    c = pt.view_as_complex(
        pt.view_as_real(num) / pt.clamp(denom[..., None], eps_scaling["mdp"])
    )

    return Y * c[..., None].conj()


def minimum_distortion_l2_phase(Y: pt.Tensor, ref: pt.Tensor) -> pt.Tensor:
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Parameters
    ----------
    Y: torch.Tensor (..., n_channels, n_frequencies, n_frames)
        The STFT data to project back on the reference signal
    ref: torch.Tensor (..., n_frequencies, n_frames)
        The reference signal
    """
    Y = Y.transpose(-3, -2)  # put Y in order (..., n_freq, n_chan, n_frames)

    num = pt.view_as_real(pt.mean(ref[..., None, :].conj() * Y, dim=-1))
    # only correct the phase
    c = pt.view_as_complex(num / pt.clamp(pt.norm(num, dim=-1, keepdim=True), min=1e-5))

    return (Y * c[..., :, :, None].conj()).transpose(-3, -2)


def minimum_distortion(
    Y: pt.Tensor,
    ref: pt.Tensor,
    p: Optional[float] = None,
    q: Optional[float] = None,
    rtol: Optional[float] = 1e-2,
    max_iter: Optional[float] = 100,
    model: Optional[pt.nn.Module] = None,
) -> pt.Tensor:
    """
    This function computes the frequency-domain filter that minimizes the sum
    of errors to a reference signal with a mixed-norm. This is a sparse version
    of the projection back that is commonly used to solve the scale ambiguity
    in BSS.

    Parameters
    ----------
    Y: array_like (n_frequencies, n_channels, n_frames)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_freq)
        The reference signal
    p: float (0 < p <= 2)
        The norm to use to measure distortion
    q: float (0 < p <= q <= 2)
        The other exponent when using a mixed norm to measure distortion
    max_iter: int, optional
        Maximum number of iterations
    rtol: float, optional
        Stop the optimization when the algorithm makes less than rtol relative progress
    model: torch.nn.Module
        An optional learnable block to replace the MM weights
    """

    # by default we do the regular minimum distortion
    if model is None and (p is None or (p == 2.0 and (q is None or p == q))):
        return minimum_distortion_l2(Y, ref)

    n_freq, n_channels, n_frames = Y.shape[-3:]

    # make stuff contiguous to make pytorch happy
    # the Y order should be (..., n_freq, n_chan, n_frames)
    Y_bak = Y  # keep reference
    Y = Y.transpose(-3, -2).contiguous()
    ref = ref.contiguous()

    c = Y.new_ones(Y.shape[:-1])

    prev_c = None

    epoch = 0
    while epoch < max_iter:

        epoch += 1

        # the current error
        error = ref[..., :, None, :] - c[..., :, :, None] * Y
        if model is not None:
            weights = model(error.transpose(-3, -2)).transpose(-3, -2)
            weights = weights[..., 0, 0, :, :]

            # let us normalize the weights along the time axis
            norm = pt.sum(weights, dim=-1, keepdim=True)
            norm = pt.clamp(norm, min=1e-5)
            weights = weights / norm
        elif q is None or p == q:
            weights = lp_norm(error, p=p)
        else:
            weights = lpq_norm(error, p=p, q=q, axis=-3)  # axis=-3 -> frequency

        # minimize
        num = pt.sum(ref[..., None, :] * Y.conj() * weights, dim=-1)
        denom = pt.sum(Y.abs().square() * weights, dim=-1)
        c = num / pt.clamp(denom, min=eps_scaling["gmdp"])

        # condition for termination
        if prev_c is None:
            prev_c = c
            continue

        # relative step length
        delta = (c - prev_c).abs().norm() / (prev_c).abs().norm()
        prev_c = c
        if delta < rtol:
            break

    return Y_bak * c[..., None].transpose(-3, -2)


def lp_norm(E: pt.Tensor, p: Optional[int] = 1) -> pt.Tensor:
    assert p > 0 and p < 2
    weights = p / pt.clamp(2.0 * E.abs().pow(2 - p), min=eps_scaling["gmdp"])
    return weights


def lpq_norm(E, p=1, q=2, axis=0) -> pt.Tensor:
    assert p > 0 and q >= p and q <= 2.0

    E = E.abs() + 1e-5
    E = E.pow(q) + 1e-5
    rn = pt.sum(E, dim=axis, keepdim=True).pow(1 - p / q)
    qfn = pt.abs(E).pow(2 - q)
    weights = p / pt.clamp(2.0 * rn * qfn, min=eps_scaling["gmdp"])
    return weights
