import torch
from torchiva.dtypes import is_complex_type, dtype_f2cpx, dtype_cpx2f
import pytest


@pytest.mark.parametrize(
    "dtype, ret",
    [
        (torch.complex64, True),
        (torch.complex128, True),
        (torch.float32, False),
        (torch.float64, False),
    ],
)
def test_is_complex_type(dtype, ret):
    x = torch.zeros(1, dtype=dtype).normal_()
    assert is_complex_type(dtype) == ret
    assert is_complex_type(x) == ret

@pytest.mark.parametrize(
    "dtype, ret_dtype",
    [
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_dtype_cpx2f(dtype, ret_dtype):
    x = torch.zeros(1, dtype=dtype).normal_()
    assert dtype_cpx2f(dtype) == ret_dtype
    assert dtype_cpx2f(x) == ret_dtype

@pytest.mark.parametrize(
    "dtype, ret_dtype",
    [
        (torch.complex64, torch.complex64),
        (torch.complex128, torch.complex128),
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ],
)
def test_dtype_cpx2f(dtype, ret_dtype):
    x = torch.zeros(1, dtype=dtype).normal_()
    assert dtype_f2cpx(dtype) == ret_dtype
    assert dtype_f2cpx(x) == ret_dtype
