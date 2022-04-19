# Copyright (c) 2022 Robin Scheibler
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
