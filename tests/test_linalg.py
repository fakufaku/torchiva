import torch
import pytest
import torchiva

tol = 1e-5


@pytest.mark.parametrize(
    "dtype",
    [torch.complex64, torch.complex128, torch.float32, torch.float64],
)
def test_eigh_2x2(dtype):

    # random PSD matrix
    A = torch.zeros((2, 10), dtype=dtype).uniform_()
    A = A @ torchiva.linalg.hermite(A)

    e1, v1 = torch.linalg.eigh(A)
    e2, v2 = torchiva.linalg.eigh_2x2(A)

    err_e = abs(e1 - e2).max()
    err_v = abs(abs(v1.T @ v2) - torch.eye(2)).max()

    assert err_e < tol
    assert err_v < tol


@pytest.mark.parametrize(
    "dtype",
    [torch.complex64, torch.complex128, torch.float32, torch.float64],
)
def test_general_eigh_2x2(dtype):

    # random PSD matrix
    A = torch.zeros((2, 10), dtype=dtype).uniform_()
    A = A @ torchiva.linalg.hermite(A)

    B = torch.zeros((2, 10), dtype=dtype).uniform_()
    B = B @ torchiva.linalg.hermite(B)

    L, V = torchiva.linalg.eigh_2x2(A, B)

    err = abs(A @ V - (B @ V) * L[None, :]).max()

    assert err < tol
