import time
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

def measure_eigh_runtime(batch, dtype):
    rep = 20

    # random PSD matrix
    A = torch.zeros((batch, 2, 10), dtype=dtype).uniform_()
    A = A @ torchiva.linalg.hermite(A)

    A = A.to(0)

    e1, v1 = torch.linalg.eigh(A)
    ts = time.perf_counter()
    for i in range(rep):
        e1, v1 = torch.linalg.eigh(A)
    runtime_torch = (time.perf_counter() - ts) / rep

    e2, v2 = torchiva.linalg.eigh_2x2(A)
    ts = time.perf_counter()
    for i in range(rep):
        e2, v2 = torchiva.linalg.eigh_2x2(A)
    runtime_torchiva = (time.perf_counter() - ts) / rep


    print(f"{batch=} {dtype=} {runtime_torch=:.6f} {runtime_torchiva=:.6f}")

if __name__ == "__main__":
    measure_eigh_runtime(1, torch.complex64)
    measure_eigh_runtime(10, torch.complex64)
    measure_eigh_runtime(100, torch.complex64)
    measure_eigh_runtime(1000, torch.complex64)
    measure_eigh_runtime(1, torch.complex128)
    measure_eigh_runtime(10, torch.complex128)
    measure_eigh_runtime(100, torch.complex128)
    measure_eigh_runtime(1000, torch.complex128)
