# Implementation of the proposed MM algorithm for DOA refinements
#
# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import abc
from enum import Enum

import torch
import torch.nn as nn
import math
import numpy as np
import pyroomacoustics as pra

from utils import cartesian_to_spherical


class SurrogateType(Enum):
    Linear = "Linear"
    Quadratic = "Quadratic"


def eigh_wrapper(V, use_cpu=True):

    if use_cpu:
        dev = V.device

        # diagonal loading factor
        # dload = 1e-5 * pt.eye(V.shape[-1], device=V.device, dtype=V.dtype)
        # V = (V + dload).cpu()
        V = V.cpu()
        e_val, e_vec = torch.linalg.eigh(V)

        return e_val.to(dev), e_vec.to(dev)
    else:
        return torch.linalg.eigh(V)


def power_mean(X, s=1.0, *args, **kwargs):

    if s != 1.0:
        return (torch.mean(X ** s, *args, **kwargs)) ** (1.0 / s)
    else:
        return torch.mean(X, *args, **kwargs)


def compute_covariance_matrix(X, mask=None):
    n_frames = X.shape[-1]
    if mask is None:
        C_hat = torch.einsum("...mft,...nft->...fmn", X, torch.conj(X)) / n_frames
    else:
        C_hat = torch.einsum("...ft,...mft,...nft->...fmn", mask, X, torch.conj(X))
    return C_hat


def cosine_cost(q, mics, wavenumbers, data_mag, data_arg, data_const, s=1.0):
    """
    Computes the value of the cost function of DOA-MM.
    Parameters
    ----------
    q: array_like, shape (..., n_dim)
        the current propagation vector estimate
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_freq)
        the wavenumbers corresponding to each data bin
    data: ndarray, shape (n_freq, n_mics)
        the data that defines the noise subspace
    s: float
        the power mean parameter
    Returns
    -------
    cost: float
        the value of the MMUSIC cost function
    """
    n_mics, n_dim = mics.shape
    n_freq = wavenumbers.shape[0]
    batch_shape = data_mag.shape[:-2]
    assert data_mag.shape == data_arg.shape
    assert data_mag.shape[:-1] == data_const.shape
    assert n_mics == data_mag.shape[-1]
    assert n_freq == data_mag.shape[-2]
    assert n_dim == q.shape[-1]
    assert q.shape[:-1] == batch_shape

    delta_t = torch.einsum(
        "f,md,...d->...fm", wavenumbers, mics, q
    )  # (..., n_freq, n_mics)
    e = data_arg - delta_t  # shape (..., n_freq, n_mics)

    ell = data_const + 2 * torch.sum(data_mag * torch.cos(e), dim=-1)

    cost = power_mean(ell, s=s, axis=-1)

    return cost


def extract_off_diagonal(X):
    """
    Parameters
    ----------
    X: array_like, shape (..., M, M)
        A multi dimensional array
    Returns
    -------
    Y: array_like, shape (..., M * (M - 1) / 2)
        The linearized entries under the main diagonal
    """
    # we need to format the sensors
    M = X.shape[-1]
    assert X.shape[-2] == M
    indices = np.arange(M)

    mask = np.ravel_multi_index(np.where(indices[:, None] > indices[None, :]), (M, M))

    new_shape = X.shape[:-2] + (X.shape[-2] * X.shape[-1],)
    X = X.reshape(new_shape)
    if X.dtype in [torch.complex64, torch.complex128]:
        X = torch.view_as_real(X)
        X = X[..., mask, :]
        X = torch.view_as_complex(X)
    else:
        X = X[..., mask]

    return X


def cosine_majorization(q, mics, wavenumbers, data_mag, data_arg, data_const, s=1.0):
    """
    Computes the auxiliary variables of the majorization of the cosine
    cost function.
    Parameters
    ----------
    q: array_like, shape (..., n_dim)
        the current propagation vector estimate
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_freq)
        the wavenumbers corresponding to each data bin
    data_mag: ndarray, shape (..., n_freq, n_mics)
        the magnitude of the data
    data_arg: ndarray, shape (..., n_freq, n_mics)
        the phase of the data
    data_const: ndarray, shape (..., n_freq, n_mics)
        the additive constant
    s: float
        the power mean parameter
    Returns
    -------
    new_data: ndarray, shape (..., n_mics)
        the auxiliary right hand side
    weights: ndarray, shape (..., n_mics)
        the new weights
    """
    n_mics, n_dim = mics.shape
    n_freq = wavenumbers.shape[0]
    batch_shape = data_mag.shape[:-2]
    assert data_mag.shape == data_arg.shape
    assert data_mag.shape[:-1] == data_const.shape
    assert n_mics == data_mag.shape[-1]
    assert n_freq == data_mag.shape[-2]
    assert n_dim == q.shape[-1]
    assert q.shape[:-1] == batch_shape

    # put into linear batch size
    data_mag = data_mag.reshape((-1, n_freq, n_mics))
    data_arg = data_arg.reshape((-1, n_freq, n_mics))
    data_const = data_const.reshape((-1, n_freq))
    q = q.reshape((-1, n_dim))

    # subtract pi to phase because this is the majorization
    data_arg = data_arg - math.pi

    # prepare the auxiliary variable
    delta_t = torch.einsum(
        "f,md,bd->bfm", wavenumbers, mics, q
    )  # (n_batch, n_freq, n_mics)
    e = data_arg - delta_t

    # compute the offset to pi
    z = torch.round(e / (2.0 * math.pi))
    zpi = 2 * z * math.pi
    phi = e - zpi

    # compute the weights
    weights = data_mag * torch.sinc(torch.clamp(phi, min=1e-5) / math.pi)
    new_data = data_arg - zpi

    # this the time-frequency bin weight corresponding to the robustifying function
    # shape (n_points)
    if s < 1.0:
        ell = data_const + 2 * torch.sum(data_mag * torch.cos(e), dim=-1)
        r = (
            (1.0 / n_freq)
            * ell ** (s - 1.0)
            / torch.mean(ell ** s, axis=-1, keepdim=True) ** (1.0 - 1.0 / s)
        )
        weights = weights * r[..., None]

    # We can reduce the number of terms by completing the squares
    weights_red = torch.einsum("bfm,f->bm", weights, wavenumbers ** 2)

    r_red = torch.einsum("bfm,bfm,f->bm", new_data, weights, wavenumbers)
    data_red = r_red / torch.clamp(weights_red, min=1e-5)

    # restore batch_size
    data_red.reshape(batch_shape + data_red.shape[1:])
    weights_red.reshape(batch_shape + data_red.shape[1:])

    return data_red, weights_red


def mm_refinement_step(
    q, mics, mics_diff, wavenumbers, data_mag, data_arg, data_const, ev_max, s=1.0
):
    """
    Perform one step of the DOA MM refinement algorithm
    Parameters
    ----------
    q: array_like, shape (n_dim,)
        the initial direction vector
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_freq)
        the wavenumbers corresponding to each data bin
    data: ndarray, shape (n_freq, n_mics)
        the phase of the measurements
    """
    n_mics, n_dim = mics.shape
    n_freq = wavenumbers.shape[0]
    batch_shape = data_mag.shape[:-2]
    assert data_mag.shape == data_arg.shape
    assert data_mag.shape[:-1] == data_const.shape
    assert n_mics == data_mag.shape[-1]
    assert n_freq == data_mag.shape[-2]
    assert n_dim == q.shape[-1]
    assert q.shape[:-1] == batch_shape

    # put into linear batch size
    data_mag = data_mag.reshape((-1, n_freq, n_mics))
    data_arg = data_arg.reshape((-1, n_freq, n_mics))
    data_const = data_const.reshape((-1, n_freq))
    q = q.reshape((-1, n_dim))

    # new_data.shape == (..., n_mics)
    # new_weights.shape == (..., n_mics)
    # the applies the cosine majorization
    new_data, new_weights = cosine_majorization(
        q, mics, wavenumbers, data_mag, data_arg, data_const, s=s
    )

    # majorization by a linear function
    C = torch.max(new_weights, dim=-1).values * ev_max  # (...,)
    y = torch.einsum("dm,bm,bm->bd", mics_diff, new_data, new_weights)
    # y = mics_diff @ (new_data * new_weights).T
    # Lq = mics_diff @ (new_weights[:, :, None] * (mics_diff.T @ qs.T))
    v1 = torch.einsum("bm,dm,bd->bm", new_weights, mics_diff, q)
    Lq = torch.einsum("bm,dm->bd", v1, mics_diff)
    # Lq = torch.einsum("em,dm,bm,bd->de", mics_diff, mics_diff, new_weights, q)

    # compute new direction
    q = y - Lq + C[:, None] * q

    # apply norm constraint
    q_norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    q = q / torch.clamp(q_norm, min=1e-5)

    # reshape batch size
    q = q.reshape(batch_shape + q.shape[-1:])

    return q


class DOAMMBase(nn.Module):
    """
    Implements the MUSCIC DOA algorithm with optimization directly on the array
    manifold using an MM algorithm
    .. note:: Run locate_sources() to apply MMUSIC
    Parameters
    ----------
    L: array_like, shape (n_dim, n_mics)
        Contains the locations of the microphones in the columns
    fs: int or float
        Sampling frequency
    nfft: int
        FFT size
    c: float, optional
        Speed of sound
    num_src: int, optional
        The number of sources to recover (default 1)
    s: float
        The exponent for the robustifying function, we expect that making beta larger
        should make the method less sensitive to outliers/noise
    n_grid: int
        The size of the grid search for initialization
    track_cost: bool
        If True, the cost function will be recorded
    init_grid: int
        Size of the grid for the rough initialization method
    verbose: bool, optional
        Whether to output intermediate result for debugging purposes
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        s=1.0,
        dim=None,
        n_grid=100,
        n_iter=5,
        freq_range_hz=None,
        z_only_positive=False,
        *args,
        **kwargs,
    ):
        """
        The init method
        """
        super().__init__()

        # microphones
        self.register_buffer("L", torch.as_tensor(L))

        
        # if dimension is unspecified infer from mic array locations
        if dim is None:
            dim = self.L.shape[0]
        
        assert self.L.shape[0] in [
            2,
            3,
        ], "Microphones should be in columns of array of size (n_dim, n_mics)"

        #assert num_src == 1, "Specialized for single source"
        

        # MM algorithm parameters
        self.s = s  # robustifying function parameter
        self.n_grid = n_grid  # initial estimation grid size
        self.n_iter = n_iter  # defualt number of iterations for MM

        # system parameters
        self.fs = fs
        self.c = c
        self.num_src = num_src
        self.nfft = nfft
        self.dim = dim

        # differential microphone locations (for x-corr measurements)
        # shape (n_dim, n_mics * (n_mics - 1) / 2)
        self.register_buffer(
            "_L_diff", extract_off_diagonal(self.L[:, :, None] - self.L[:, None, :]),
        )

        # for the linear type algorithm, we need
        self.register_buffer("_L_diff2", self._L_diff @ self._L_diff.T)
        self.register_buffer("ev_max", torch.max(torch.linalg.eigvalsh(self._L_diff2)))

        # create the grid for initial estimation
        if self.dim == 2:
            self.grid = pra.doa.grid.GridCircle(n_points=self.n_grid)
            self.register_buffer(
                "grid_cartesian", torch.as_tensor(self.grid.spherical, dtype=self.L.dtype)
            )
        elif self.dim == 3:
            self.grid = pra.doa.grid.GridSphere(n_points=self.n_grid)
            self.register_buffer(
                "grid_cartesian", torch.as_tensor(self.grid.cartesian, dtype=self.L.dtype)
            )
        else:
            raise NotImplementedError("Only 2D and 3D arrays are supported")
        #self.register_buffer(
        #    "grid_cartesian", torch.as_tensor(self.grid.cartesian, dtype=self.L.dtype)
        #)

        # change gridsphere to hemisphere
        if z_only_positive and self.dim==3:
            self.grid_cartesian = self.grid_cartesian[:,self.n_grid//2:]
            self.grid = pra.doa.grid.GridSphere(n_points=self.n_grid//2)


        self.n_freq = self.nfft // 2 + 1
        self.register_buffer("freq_hz", torch.arange(self.n_freq) / self.nfft * self.fs)

        if freq_range_hz is None:
            self.freq_range_hz = [self.freq_hz[1], self.freq_hz[-1]]
            self.freq_select = self.freq_hz > 0
        else:
            self.freq_range_hz = freq_range_hz
            self.freq_select = (freq_range_hz[0] <= self.freq_hz) & (
                self.freq_hz <= freq_range_hz[1]
            )

        # create the mode vectors (aka steering vectors)
        wavenum = 2.0 * math.pi * self.freq_hz[self.freq_select] / c
        wavenum=wavenum.to(self.L.device)
        self.grid_cartesian=self.grid_cartesian.to(self.L.device)
        self.register_buffer(
            "mode_vec",
            torch.exp(
                1j * torch.einsum("f,dm,dg->gfm", wavenum, self.L, self.grid_cartesian)
            ),
        )

    @abc.abstractmethod
    def _compute_cost_matrix(self, X):
        raise NotImplementedError

    def grid_search(self, R):
        """
        Evaluate the cost function and saves the result in the grid values
        Parameters
        ----------
        R: shape (n_freq_selected, n_mics, n_mics)
            Array containing the covariance matrices
        select: shape (n_freq,) or (n_freq_selected)
            A boolean array to select which frequency bins to use, or a list of indices
        Returns
        -------
        The cost function on the grid
        """

    
        gvec = self.mode_vec

        pwr = torch.einsum("gfm,...fmn,gfn->...gf", torch.conj(gvec), R, gvec)
        pwr = pwr.real
        cost = power_mean(pwr, dim=-1)  # shape == (..., n_grid,)

        # use the grid to find the minimums of the cost
        if self.num_src == 1:
            weights = torch.softmax(1.0 / torch.clamp(cost, min=1e-6), dim=-1)
            q0 = torch.einsum("...g,dg->...d", weights, self.grid_cartesian)
            # src_idx = torch.min(cost, dim=-1).indices
            # q0 = self.grid_cartesian.T[src_idx, :]
            q0 = q0[..., None, :]
            q0 = q0 / torch.clamp(torch.linalg.norm(q0, dim=-1, keepdim=True), min=1e-6)
            # import pdb
            # pdb.set_trace()
        else:
            batch_shape = cost.shape[:-1]
            cost = cost.reshape((-1, cost.shape[-1]))

            q0_list = []
            for b in range(cost.shape[0]):
                self.grid.values = 1.0 / cost[b].detach().cpu().numpy()
                src_idx = self.grid.find_peaks(k=self.num_src)
                q0_list.append(self.grid_cartesian.T[None, src_idx, :])

            q0 = torch.cat(q0_list, dim=0)
            q0 = q0.reshape(batch_shape + q0.shape[-2:])

        return q0

    def prepare_mm_data(self, R):

        # the wavenumbers (n_freq * n_frames)
        wavenumbers = 2 * math.pi * self.freq_hz[self.freq_select] / self.c
        wavenumbers = wavenumbers.to(self.L.device)

        # For x-corr measurements, we consider differences of microphones as sensors
        # n_mics = n_channels * (n_channels - 1) / 2
        n_mics = self._L_diff.shape[1]

        # shape (n_mics, n_dim)
        mics = self._L_diff.T

        # shape (n_freq, n_mics)
        # here the minus sign is to account for the change of maximization
        # into minimization
        data = extract_off_diagonal(R)

        # separate into magnitude and phase
        data_mag = torch.abs(data)
        data_arg = torch.angle(data + 1e-5)
        data_const = torch.abs(torch.einsum("...mm->...", R))

        return data_mag, data_arg, data_const, mics, wavenumbers

    def refine(self, q, R, n_iter=None, verbose=False, return_cost=False):

        if n_iter is None:
            n_iter = self.n_iter

        # restore
        data_mag, data_arg, data_const, mics, wavenumbers = self.prepare_mm_data(R)

        # Run the DOA algoritm
        cost = []

        for epoch in range(n_iter):

            q = mm_refinement_step(
                q,
                mics,
                self._L_diff,
                wavenumbers,
                data_mag,
                data_arg,
                data_const,
                self.ev_max,
                s=self.s,
            )

            if verbose:
                doa, r = cartesian_to_spherical(q[None, :].T.detach().numpy())
                c = cosine_cost(
                    q, mics, wavenumbers, data_mag, data_arg, data_const, s=self.s
                )
                print(f"Epoch {epoch}")
                print(
                    f"  colatitude={np.degrees(doa[0, :])}\n"
                    f"  azimuth=   {np.degrees(doa[1, :])}\n"
                    f"  cost=      {c}\n"
                )

            if return_cost:
                c = cosine_cost(
                    q, mics, wavenumbers, data_mag, data_arg, data_const, s=self.s
                )
                cost.append(c)
                if verbose:
                    print(f"  cost: {c}")

        if return_cost:
            return q, cost
        else:
            return q

    def forward(self, X, mask=None, n_iter=None, verbose=False, return_cost=False):
        """
        Process the input data and computes the DOAs
        Parameters
        ----------
        X: array_like, shape (..., n_channels, n_freq_selected, n_frames)
            The multichannel STFT of the microphone signals
            Set of signals in the frequency (RFFT) domain for current
            frame. Size should be M x F x S, where M should correspond to the
            number of microphones, F to nfft/2+1, and S to the number of snapshots
            (user-defined). It is recommended to have S >> M.
        mask: array_like, shape (..., n_freq_selected, n_frames)
            A time-frequency mask that we apply prior to performing the computations
        n_iter: int
            Number of iterations of MM
        verbose: bool
            Print evolution of MM algorithm over iterations
        Returns
        -------
        q: tensor (..., n_src, n_dim)
            The unit norm vectors corresponding to the source directions
        """
        X = X[...,self.freq_select,:]

        if n_iter is None:
            n_iter = self.n_iter

        batch_shape = X.shape[:-3]
        n_dim = self.dim
        n_chan, n_freq, n_frames = X.shape[-3:]

        assert self.L.shape[1] == n_chan

        X = X.reshape((-1, n_chan, n_freq, n_frames))
        if mask is not None:
            mask = mask.reshape((-1,) + mask.shape[-2:])

        # Compute the matrix from the objective
        R = self._compute_cost_matrix(X, mask=mask)

        # Initialize by grid search
        q0 = self.grid_search(R)

        # STEP 3: refinement via the MM algorithm
        q_lst = []
        cost = []
        for k in range(self.num_src):
            if return_cost:
                qk, costk = self.refine(q0[..., k, :], R, n_iter=n_iter, verbose=verbose, return_cost=True)
                cost.append(costk[-1])
            else:
                qk = self.refine(q0[..., k, :], R, n_iter=n_iter, verbose=verbose)
            q_lst.append(qk[..., None, :])
            
        q = torch.cat(q_lst, dim=-2)

        q = q.reshape(batch_shape + q.shape[-2:])
        if return_cost:
            return q, cost
        else:
            return q


class MMMUSIC(DOAMMBase):
    def _compute_cost_matrix(self, X, mask=None):

        # STEP 1: Classical MUSIC
        # compute the covariance matrices (also needed for STEP 2)
        # shape (n_freq, n_mic, n_mic)
        C_hat = compute_covariance_matrix(X, mask=mask)

        # subspace decomposition
        w, E = eigh_wrapper(C_hat)
        En = E[..., : -self.num_src]  # keep the noise subspace basis
        R_music = torch.einsum("...mn,...kn->...mk", En, torch.conj(En))

        return R_music


class MMSRP(DOAMMBase):
    """
    Implements the SRP-PHAT DOA algorithm with optimization directly on the array
    manifold using an MM algorithm
    """

    def _compute_cost_matrix(self, X, mask=None):
        # STEP 1: Classic SRP-PHAT

        # apply PHAT weighting
        absX = torch.abs(X)
        absX[absX < 1e-15] = 1e-15
        pX = X / absX

        # compute the covariance matrices (also needed for STEP 2)
        # shape (n_freq, n_mic, n_mic)
        C_hat = compute_covariance_matrix(pX, mask=mask)

        # We transform the covariance matrices to make SRP-PHAT a minimization problem
        # the offset l_max is added so that the cost function obtained stays larger than zero
        l_max = torch.max(torch.sum(torch.abs(C_hat), dim=-1), dim=-1).values
        M = self.L.shape[1]
        C_hat = (
            M
            * l_max[..., None, None]
            * torch.eye(C_hat.shape[-1], device=l_max.device)[None, None, :, :]
            - C_hat
        )
        return C_hat