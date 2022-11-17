import abc
import math
from enum import Enum
from typing import Optional

import torch as pt

from ..linalg import mag_sq
from .parameters import eps_models

try:
    from math import prod
except ImportError:

    def prod(*args):
        out = args[0]
        for a in args[1:]:
            out *= a
        return out


class LaplaceModel(pt.nn.Module):
    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self._eps = eps if eps is not None else eps_models["laplace"]

        # just so that training works
        self.fake = pt.nn.Parameter(pt.zeros(1))

    def cost(self, Y):
        return pt.linalg.norm(Y, dim=-2).mean(dim=-1).sum(dim=-1)

    def forward(self, X: pt.Tensor):
        # sum power over frequencies
        if X.dtype in [pt.complex64, pt.complex128]:
            mag_sq = X.real.square() + X.imag.square()
        else:
            mag_sq = X.square()
        denom = 2.0 * pt.sqrt(mag_sq.sum(dim=-2, keepdim=True))
        r = pt.broadcast_to(denom, X.shape)

        r_inv = 1.0 / pt.clamp(r, min=self._eps)
        return r_inv


class GaussModel(pt.nn.Module):
    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self._eps = eps if eps is not None else eps_models["gauss"]

        # just so that training works
        self.fake = pt.nn.Parameter(pt.zeros(1))

    def forward(self, X: pt.Tensor):
        # sum power over frequencies
        if X.dtype in [pt.complex64, pt.complex128]:
            mag_sq = X.real.square() + X.imag.square()
        else:
            mag_sq = X.square()
        denom = mag_sq.mean(dim=-2)
        _, r = pt.broadcast_tensors(X, denom[..., None, :])

        r_inv = 1.0 / pt.clamp(r, min=self._eps)
        return r_inv


class NMFModel(pt.nn.Module):
    def __init__(self, n_basis: Optional[int] = 2, eps: Optional[float] = None):
        super().__init__()
        self._eps = eps if eps is not None else eps_models["nmf"]
        self.n_basis = n_basis

        self.reset()

    def reset(self):
        # we set this to true when we fix the dimensions of the tensors
        self._is_initialized = False

        # the number of basis functions for NMF, i.e. the rank of the matrix
        self.n_freq = None
        self.n_frames = None
        self.n_batch = None

        # we do lazy initialization so that we can infer the size at
        # runtime
        self.T = None
        self.V = None
        self.iR = None

    def _init_state(self, P):

        if not self._is_initialized:
            self._is_initialized = True

            # save all the dimensions
            self.n_batch, self.n_freq, self.n_frames = P.shape

            # initialize with uniform values between 0.1 and 0.9
            self.T = P.new_zeros((self.n_batch, self.n_freq, self.n_basis)).uniform_()
            self.T = 0.9 * self.T + 0.1
            self.V = P.new_zeros((self.n_batch, self.n_frames, self.n_basis)).uniform_()
            self.V = 0.9 * self.V + 0.1

            # initialize the estimate matrix
            self._recompute_estimate()

    def _recompute_estimate(self):
        self.R = pt.bmm(self.T, self.V.transpose(-2, -1))
        self.R = pt.clamp(self.R, min=self._eps)
        self.iR = pt.reciprocal(self.R)

    def forward(self, X: pt.Tensor):
        """
        Parameters
        ----------
        X: torch.Tensor, shape (..., n_freq, n_frames)
            The input signals in the STFT domain
        """

        # flatten the batch dimensions
        batch_shape = X.shape[:-2]
        n_batch = prod(batch_shape)
        n_freq, n_frames = X.shape[-2:]
        X = X.reshape((-1, n_freq, n_frames))

        # squared magnitude of the spectrums
        P = mag_sq(X)

        # this function will make sure that all the tensors are initialized
        self._init_state(P)

        # The basis matrix update
        self.T = self.T * pt.sqrt(
            pt.bmm(P * pt.square(self.iR), self.V) / pt.bmm(self.iR, self.V)
        )
        self.T = pt.clamp(self.T, min=self._eps)

        # recompute R = dot(T, V) and its reciprocal
        self._recompute_estimate()

        # The activation matrix update
        iR_T = self.iR.transpose(-2, -1)
        P_T = P.transpose(-2, -1)
        self.V = self.V * pt.sqrt(
            pt.bmm(P_T * pt.square(iR_T), self.T) / pt.bmm(iR_T, self.T)
        )
        self.V = pt.clamp(self.V, min=self._eps)

        # recompute R = dot(T, V) and its reciprocal
        self._recompute_estimate()

        # restore the batch shape before returning
        iR = self.iR.reshape(batch_shape + (n_freq, n_frames))

        return iR
