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