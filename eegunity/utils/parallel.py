"""Parallel execution utilities for EEGUnity batch processing.

This module provides the central dispatcher used by
:meth:`~eegunity.modules.batch.eeg_batch.EEGBatch.batch_process`.
Two execution backends are available in addition to the default
sequential mode:

Thread mode (``'thread'``)
    Uses :class:`~concurrent.futures.ThreadPoolExecutor`.  The CPython
    GIL is released during blocking I/O, so this mode is well-suited to
    network- or disk-bound workloads (e.g. reading files from NFS or
    issuing ``os.stat()`` calls over a network filesystem).

Process mode (``'process'``)
    Uses :class:`~concurrent.futures.ProcessPoolExecutor`.  Each worker
    runs in a separate OS process with its own GIL, enabling genuine CPU
    parallelism.  The callable is serialised with *cloudpickle*, which
    supports closures and locally-defined functions that the built-in
    :mod:`pickle` cannot handle.  This mode is suited to CPU-intensive
    signal-processing operations such as filtering, ICA, or resampling.
"""

import cloudpickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple

_WORKER_FUNC = None


def _init_process_worker(pickled_func: bytes) -> None:
    """Unpickle ``app_func`` once when a worker process starts."""
    global _WORKER_FUNC
    _WORKER_FUNC = cloudpickle.loads(pickled_func)


def _process_task(task: Tuple) -> Any:
    """Invoke the worker-local function on one applicable task."""
    _, row, should_apply = task
    if not should_apply:
        return None
    return _WORKER_FUNC(row)


def parallel_execute(
    tasks: List[Tuple],
    app_func: Callable,
    is_patch: bool,
    result_type: Optional[str],
    execution_mode: Optional[str],
    num_workers: int,
) -> List[Any]:
    """Apply *app_func* to each eligible task row and return an ordered result list.

    This is the central dispatcher for
    :meth:`~eegunity.modules.batch.eeg_batch.EEGBatch.batch_process`.
    It selects the execution backend based on *execution_mode* and
    *num_workers*, then runs *app_func* on every task whose *should_apply*
    flag is ``True``.

    Parameters
    ----------
    tasks : list of tuple
        Sequence of ``(index, row, should_apply)`` triples where *index*
        is the locator row index, *row* is a ``pandas.Series``, and
        *should_apply* is the ``bool`` result of ``con_func(row)``.
    app_func : callable
        Function to apply to each row when *should_apply* is ``True``.
        Must accept a single positional argument (the ``pandas.Series``
        row).  May be a closure or a locally-defined function.
    is_patch : bool
        When ``True``, rows for which *should_apply* is ``False``
        contribute a placeholder value to the output so that the result
        list length matches the number of tasks.  When ``False`` those
        rows are represented by ``None`` in the returned list (callers
        that pass ``is_patch=False`` typically filter ``None`` values
        afterwards).
    result_type : {'series', 'value', None}
        Controls which placeholder is inserted for skipped rows.  When
        ``'series'`` and *is_patch* is ``True``, the original *row* is
        used as placeholder, preserving the full ``pandas.Series``; in
        all other cases ``None`` is used.
    execution_mode : {'thread', 'process', None}
        Selects the concurrency backend:

        ``'thread'``
            Concurrent execution via
            :class:`~concurrent.futures.ThreadPoolExecutor`.  The Python
            GIL is released during blocking I/O, making this mode
            effective for network- or disk-bound workloads.  *num_workers*
            controls the thread-pool size.

        ``'process'``
            Concurrent execution via
            :class:`~concurrent.futures.ProcessPoolExecutor`.  Each
            worker unpickles ``app_func`` once in its initializer, then
            runs in a separate OS process.  *app_func* is serialised with
            *cloudpickle* so closures and locally-defined functions are
            supported.  *num_workers* controls the process-pool size.

        ``None``
            Sequential execution on the calling thread, regardless of
            *num_workers*.  Use this mode for lightweight operations or
            whenever concurrent access to shared state (e.g. an open
            HDF5 file handle) must be avoided.

    num_workers : int
        Maximum number of concurrent workers for thread or process pools.
        When ``<= 0``, execution always falls back to the sequential path
        even if *execution_mode* is ``'thread'`` or ``'process'``.

    Returns
    -------
    list
        Ordered result list with one entry per task.  Entries for skipped
        rows (``should_apply=False``) contain the placeholder value
        described above.

    Raises
    ------
    ValueError
        If *execution_mode* is not one of ``'thread'``, ``'process'``,
        or ``None``.

    Examples
    --------
    >>> tasks = [(0, {'x': 1}, True), (1, {'x': 2}, False)]
    >>> parallel_execute(  # doctest: +SKIP
    ...     tasks=tasks,
    ...     app_func=lambda row: row['x'] * 2,
    ...     is_patch=False,
    ...     result_type='value',
    ...     execution_mode='thread',
    ...     num_workers=2,
    ... )
    """
    if execution_mode not in ('thread', 'process', None):
        raise ValueError(
            f"execution_mode must be 'thread', 'process', or None; "
            f"got {execution_mode!r}."
        )

    def _placeholder(row: Any) -> Any:
        """Return the appropriate fill value for a skipped row."""
        if is_patch and result_type == 'series':
            return row
        return None

    # ------------------------------------------------------------------
    # Sequential path - execution_mode is None, or num_workers <= 0
    # ------------------------------------------------------------------
    if execution_mode is None or num_workers <= 0:
        results = []
        for _, row, should_apply in tasks:
            results.append(app_func(row) if should_apply else _placeholder(row))
        return results

    # ------------------------------------------------------------------
    # Thread path
    # ------------------------------------------------------------------
    if execution_mode == 'thread':
        def _thread_worker(task: Tuple) -> Any:
            """Invoke app_func or return a placeholder for a single task."""
            _, row, should_apply = task
            return app_func(row) if should_apply else _placeholder(row)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(_thread_worker, tasks))

    # ------------------------------------------------------------------
    # Process path
    # ------------------------------------------------------------------
    # Unpickle app_func once per worker instead of once per row.
    pickled_func = cloudpickle.dumps(app_func)

    results: List[Any] = [_placeholder(row) for _, row, _ in tasks]

    applicable = [
        (original_idx, task)
        for original_idx, task in enumerate(tasks)
        if task[2]  # should_apply
    ]

    if applicable:
        applicable_tasks = [task for _, task in applicable]
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_process_worker,
            initargs=(pickled_func,),
        ) as executor:
            computed = list(executor.map(_process_task, applicable_tasks))
        for (original_idx, _), value in zip(applicable, computed):
            results[original_idx] = value

    return results
