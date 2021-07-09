from typing import Union

import torch as pt


def is_complex_type(t: Union[pt.dtype, pt.Tensor]) -> bool:

    if isinstance(t, pt.Tensor):
        t = t.dtype

    return t in [pt.complex64, pt.complex128]


def dtype_f2cpx(t: Union[pt.dtype, pt.Tensor]) -> pt.dtype:

    if isinstance(t, pt.Tensor):
        t = t.dtype

    _d = {
        pt.float32: pt.complex64,
        pt.float64: pt.complex128,
        pt.complex64: pt.complex64,
        pt.complex128: pt.complex128,
    }

    assert t in _d.keys()

    return _d[t]


def dtype_cpx2f(t: Union[pt.dtype, pt.Tensor]) -> pt.dtype:

    if isinstance(t, pt.Tensor):
        t = t.dtype

    _d = {
        pt.complex64: pt.float32,
        pt.complex128: pt.float64,
        pt.float32: pt.float32,
        pt.float64: pt.float64,
    }

    assert t in _d.keys()

    return _d[t]
