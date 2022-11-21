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

import torch


def fftconvolve(x1, x2, mode="full", dim=-1):
    """
    Simple function for computing the convolution of x1 with x2 via frequency domain
    using the FFT.  We do not implement overlap add yet.
    Parameters
    ----------
    x1: Tensor (..., n_samples_1)
        The first array
    x2: Tensor (..., n_samples_2)
        The second array
    mode: str
        The truncation mode
    """

    x1 = x1.transpose(dim, -1)
    x2 = x2.transpose(dim, -1)

    n1 = x1.shape[-1]
    n2 = x2.shape[-1]
    n = n1 + n2 - 1  # the (minimum) FFT size

    # Go to frequency domain
    X1 = torch.fft.rfft(x1, dim=-1, n=n)  # 1D real FFT
    X2 = torch.fft.rfft(x2, dim=-1, n=n)  # 1D real FFT

    # go back to time domain
    y = torch.fft.irfft(X1 * X2, dim=-1, n=n)

    if mode == "full":
        startind = 0
        endind = n

    elif mode == "same":
        retlen = n1
        startind = (n - retlen) // 2
        endind = startind + retlen

    elif mode == "valid":
        valid_len = max(n1, n2) - min(n1, n2) + 1
        startind = (n - valid_len) // 2
        endind = startind + valid_len

    else:
        raise ValueError("Acceptable mode flags are 'full', 'same', or 'valid'")

    y = y[..., startind:endind].transpose(dim, -1)

    return y


if __name__ == "__main__":

    import numpy as np
    from scipy.signal import fftconvolve as fftconvolve2

    x1 = np.random.randn(1000)
    x2 = np.random.randn(100)

    for mode in ["full", "same", "valid"]:
        y1 = fftconvolve(torch.from_numpy(x1), torch.from_numpy(x2)).numpy()
        y2 = fftconvolve2(x1, x2)
        print(np.linalg.norm(y1 - y2))
