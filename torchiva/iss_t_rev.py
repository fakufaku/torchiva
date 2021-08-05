# Copyright (c) 2021 Robin Scheibler
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
"""
Joint Dereverberation and Blind Source Separation with Itarative Source Steering
================================================================================

Online implementation of the algorithm presented in [1]_.

References
----------
.. [1] T. Nakashima, R. Scheibler, M. Togami, and N. Ono,
    JOINT DEREVERBERATION AND SEPARATION WITH ITERATIVE SOURCE STEERING,
    ICASSP, 2021, https://arxiv.org/pdf/2102.06322.pdf.
"""
from typing import Optional, List, NoReturn, Callable
import torch

from .linalg import hankel_view, mag_sq, divide, multiply
from .models import LaplaceModel
from .parameters import eps_models

from .auxiva_t_iss import (
    iss_block_update_type_1,
    iss_block_update_type_2,
    projection_back,
    iss_updates_with_H,
    demix_derev,
)


def enable_req_grad(*args):
    for arg in args:
        if arg is not None:
            arg.requires_grad_()


def detach(*args):
    for arg in args:
        if arg is not None:
            arg.detach_()


def zero_gradient(*args):
    for arg in args:
        if arg.grad is not None:
            arg.graad.zero_()


def iss_one_iter(X, X_bar, W, H, model):
    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(X)

    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
    g_sqrt = torch.sqrt(g)
    X = divide(X, g_sqrt)
    W = divide(W, g_sqrt)
    H = divide(H, g_sqrt[..., None])
    weights = weights * g

    # Iterative Source Steering updates
    X, W, H = iss_updates_with_H(X, X_bar, W, H, weights)

    return X, W, H


class ISS_T_Rev_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        model: Callable,
        n_iter: int,
        n_taps: int,
        n_delay: int,
        proj_back: bool,
        eps: float,
        checkpoints_iter: List[int],
        X: torch.Tensor,
        *model_params,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X: torch.Tensor, (..., n_channels, n_frequencies, n_frames)
            The input signal
        n_iter: int, optional
            The number of iterations
        proj_back:
            Flag that indicates if we want to restore the scale
            of the signal by projection back

        Returns
        -------
        Y: torch.Tensor, (..., n_channels, n_frequencies, n_frames)
            The separated and dereverberated signal
        """
        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        ctx.save_for_backward(X, *model_params)

        if model is None:
            model = LaplaceModel()

        if eps is None:
            eps = eps_models["laplace"]

        checkpoints_list = []

        Y = X.clone()

        # construct the matrix containing the delayed versions of the input signal
        # we use stride tricks to avoid extra memory requirements
        # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
        X_pad = torch.nn.functional.pad(X, (n_taps + n_delay, 0))
        X_hankel = hankel_view(X_pad, n_taps + n_delay + 1)
        X_bar = X_hankel[..., : -n_delay - 1, :]  # shape (c, f, t, b)

        # the demixing matrix
        W = Y.new_zeros((n_iter + 1,) + batch_shape + (n_chan, n_freq, n_chan))
        eye = torch.eye(n_chan).type_as(W)
        W[0] = eye[:, None, :]

        # the dereverberation filters
        H = Y.new_zeros((n_iter + 1,) + batch_shape + (n_chan, n_freq, n_chan, n_taps))

        with torch.no_grad():
            for epoch in range(n_iter):

                if checkpoints_iter is not None and epoch in checkpoints_iter:
                    checkpoints_list.append(Y)

                Y, W[epoch + 1], H[epoch + 1] = iss_one_iter(
                    Y, X_bar, W[epoch], H[epoch], model
                )

                # we recompute Y from W and H to avoid numerical errors in backward
                # Y = demix_derev(X, X_bar, W[epoch + 1], H[epoch + 1])

            # projection back
            if proj_back:
                Y, a = projection_back(Y, W[-1], eps=eps)
            else:
                a = None

        # here we do the bookkeeping for the backward function
        ctx.extra_args = (model, Y, X_bar, W, H, a, n_iter, proj_back, eps)

        # we can detach the computation graph from Y
        Y.detach_()

        return Y

    @staticmethod
    def backward(ctx, grad_output):

        X, *model_params = ctx.saved_tensors
        model, Y, X_bar, W, H, a, n_iter, proj_back, eps = ctx.extra_args
        grad_X = None

        for p in model_params:
            if p.grad is not None:
                p.grad.zero_()

        grad_model_params = [None for p in model_params]

        enable_req_grad(*model_params)

        if proj_back:
            with torch.no_grad():
                Y = demix_derev(X, X_bar, W[-1], H[-1])

            detach(Y)

            with torch.enable_grad():
                enable_req_grad(Y)
                zero_gradient(Y)

                Y2, _ = projection_back(Y, W[-1], eps=eps)

                gradients = torch.autograd.grad(
                    outputs=Y2,
                    inputs=[Y],
                    grad_outputs=grad_output,
                    allow_unused=True,
                )

                grad_output = [gradients[0]]

                for i, grad in enumerate(gradients[1:]):
                    if grad_model_params[i] is None:
                        grad_model_params[i] = grad
                    else:
                        if grad is not None:
                            grad_model_params[i] += grad

            # detach(Y, *model_params)

        for epoch in range(1, n_iter + 1):

            # reverse the separation
            with torch.no_grad():
                Y = demix_derev(X, X_bar, W[-epoch - 1], H[-epoch - 1])

            # compute the gradient with one forward and backward pass
            with torch.enable_grad():
                enable_req_grad(Y, *model_params)
                zero_gradient(Y, *model_params)

                Y2, *_ = iss_one_iter(Y, X_bar, W[-epoch - 1], H[-epoch - 1], model)

                gradients = torch.autograd.grad(
                    outputs=Y2,
                    inputs=[Y] + model_params,
                    grad_outputs=grad_output,
                    allow_unused=True,
                )

                grad_output = [gradients[0]]

                for i, grad in enumerate(gradients[1:]):
                    if grad_model_params[i] is None:
                        grad_model_params[i] = grad
                    else:
                        if grad is not None:
                            grad_model_params[i] += grad

        if ctx.needs_input_grad[-len(model_params) - 1]:
            grad_X = grad_output[0]

        grad_outputs = [None] * 7
        grad_outputs += [grad_X]
        grad_outputs += [
            grad if needed else None
            for (grad, needed) in zip(
                grad_model_params, ctx.needs_input_grad[-len(model_params) :]
            )
        ]

        return tuple(grad_outputs)


def iss_t_rev(
    X: torch.Tensor,
    model: Callable = None,
    n_iter: Optional[int] = 10,
    n_taps: Optional[int] = 0,
    n_delay: Optional[int] = 0,
    proj_back: Optional[bool] = True,
    eps: Optional[float] = None,
    checkpoints_iter: Optional[List[int]] = None,
) -> torch.Tensor:

    model_params = [p for p in model.parameters()]

    return ISS_T_Rev_Function.apply(
        model,
        n_iter,
        n_taps,
        n_delay,
        proj_back,
        eps,
        checkpoints_iter,
        X,
        *model_params,
    )
