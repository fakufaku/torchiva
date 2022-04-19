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

import multiprocessing as mp
import time
import traceback

from tqdm import tqdm


class ProcessingPool:
    def __init__(self, *args, **kwargs):
        self.results = []
        self.errors = []
        self._pool_args = args
        self._pool_kwargs = kwargs

    def __enter__(self):
        self.pool = mp.Pool(*self._pool_args, **self._pool_kwargs)
        self._n_tasks_done = 0
        self._n_tasks = 0
        return self

    def __exit__(self, type, value, traceback):
        self.pool.close()

    def _callback(self, x):
        self.results.append(x)
        self._n_tasks_done += 1
        if hasattr(self, "progress_bar"):
            self.progress_bar.update(1)

    def _error_callback(self, e):
        self.errors.append(e)
        self._n_tasks_done += 1
        traceback.print_exception(type(e), e, e.__traceback__)
        if hasattr(self, "progress_bar"):
            self.progress_bar.update(1)

    def push(self, func, args):
        self._n_tasks += 1
        self.pool.apply_async(
            func,
            args=args,
            callback=self._callback,
            error_callback=self._error_callback,
        )

    @property
    def busy(self):
        return self._n_tasks_done < self._n_tasks

    def wait_results(self, progress_bar=False):
        if progress_bar:
            self.progress_bar = tqdm(total=self._n_tasks)
            self.progress_bar.update(0)

        while self.busy:
            time.sleep(0.1)
            continue

        if hasattr(self, "progress_bar"):
            self.progress_bar.close()

        return self.results, self.errors
