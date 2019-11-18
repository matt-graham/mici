"""Progress bar classes for tracking progress of chains."""

import abc
import html
import sys
from timeit import default_timer as timer

try:
    from IPython import get_ipython
    from IPython.display import display as ipython_display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
try:
    import tqdm.auto as tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _in_zmq_interactive_shell():
    """Check if in interactive ZMQ shell which supports updateable displays"""
    if not IPYTHON_AVAILABLE:
        return False
    else:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True
            elif shell == 'TerminalInteractiveShell':
                return False
            else:
                return False
        except NameError:
            return False


def _format_time(total_seconds):
    """Format a time interval in seconds as a colon-delimited string [h:]m:s"""
    total_mins, seconds = divmod(int(total_seconds), 60)
    hours, mins = divmod(total_mins, 60)
    if hours != 0:
        return f'{hours:d}:{mins:02d}:{seconds:02d}'
    else:
        return f'{mins:02d}:{seconds:02d}'


def _update_stats_running_means(iter, means, new_vals):
    """Update dictionary of running statistics means with latest values."""
    if iter == 1:
        means.update({key: float(val) for key, val in new_vals.items()})
    else:
        for key, val in new_vals.items():
            means[key] += (float(val) - means[key]) / iter


class BaseProgressBar(abc.ABC):
    """Base class defining expected interface for progress bars."""

    def __init__(self, n_iter, description, position):
        """
        Args:
            n_iter (int): Number of iterations to iterate over.
            description (None or str): Description of task to prefix progress
                bar with.
            position (Tuple[int, int]): Tuple specifying position of progress
                bar within a sequence with first entry corresponding to
                zero-indexed position and the second entry the total number of
                progress bars.
        """
        assert isinstance(n_iter, int) and n_iter > 0, (
            'n_iter must be a positive integer')
        self._n_iter = n_iter
        self._description = description
        self._position = position

    def __iter__(self):
        for iter in range(self._n_iter):
            iter_dict = {}
            yield iter, iter_dict
            self.update(iter + 1, iter_dict, refresh=True)

    def __len__(self):
        return self._n_iter

    @abc.abstractmethod
    def update(self, iter, iter_dict, refresh=True):
        """Update progress bar state.

        Args:
            iter (int): New value for iteration counter.
            iter_dict (None or Dict[str, float]): Dictionary of iteration
                statistics key-value pairs to use to update postfix stats.
            refresh (bool): Whether to refresh display(s).
        """

    @abc.abstractmethod
    def __enter__(self):
        """Set up progress bar and any associated resource."""

    @abc.abstractmethod
    def __exit__(self, *args):
        """Close down progress bar and any associated resources."""


class ProgressBar(BaseProgressBar):
    """Iterable object for tracking progress of an iterative task.

    Implements both string and HTML representations to allow richer
    display in interfaces which support HTML output, for example Jupyter
    notebooks or interactive terminals.
    """

    GLYPHS = ' ▏▎▍▌▋▊▉█'
    """Characters used to create string representation of progress bar."""

    def __init__(self, n_iter, description=None, position=(0, 1),
                 displays=None, n_col=10, unit='it'):
        """
        Args:
            n_iter (int): Number of iterations to iterate over.
            description (None or str): Description of task to prefix progress
                bar with.
            position (Tuple[int, int]): Tuple specifying position of progress
                bar within a sequence with first entry corresponding to
                zero-indexed position and the second entry the total number of
                progress bars.
            displays (None or List[object]): List of objects to use to display
                visual representation(s) of progress bar. Each object much have
                an `update` method which will be passed a single argument
                corresponding to the current progress bar.
            n_col (int): Number of columns (characters) to use in string
                representation of progress bar.
            unit (str): String describing unit of per-iteration tasks.
        """
        super().__init__(n_iter, description, position)
        self._n_col = n_col
        self._unit = unit
        self._counter = 0
        self._active = False
        self._start_time = None
        self._elapsed_time = 0
        self._stats_dict = {}
        self._displays = displays

    @property
    def n_iter(self):
        """Total number of iterations to complete."""
        return self._n_iter

    @property
    def description(self):
        """"Description of task being tracked."""
        return self._description

    @property
    def counter(self):
        """Progress iteration count."""
        return self._counter

    @counter.setter
    def counter(self, value):
        self._counter = max(0, min(value, self.n_iter))

    @property
    def prop_complete(self):
        """Proportion complete (float value in [0, 1])."""
        return self.counter / self.n_iter

    @property
    def perc_complete(self):
        """Percentage complete formatted as string."""
        return f'{int(self.prop_complete * 100):3d}%'

    @property
    def elapsed_time(self):
        """Elapsed time formatted as string."""
        return _format_time(self._elapsed_time)

    @property
    def iter_rate(self):
        """Mean iteration rate if ≥ 1 `it/s` or reciprocal `s/it` as string."""
        if self.prop_complete == 0:
            return '?'
        else:
            mean_time = self._elapsed_time / self.counter
            return (
                f'{mean_time:.2f}s/{self._unit}' if mean_time > 1
                else f'{1/mean_time:.2f}{self._unit}/s')

    @property
    def est_remaining_time(self):
        """Estimated remaining time to completion formatted as string."""
        if self.prop_complete == 0:
            return '?'
        else:
            return _format_time(
                (1 / self.prop_complete - 1) * self._elapsed_time)

    @property
    def n_block_filled(self):
        """Number of filled blocks in progress bar."""
        return int(self._n_col * self.prop_complete)

    @property
    def n_block_empty(self):
        """Number of empty blocks in progress bar."""
        return self._n_col - self.n_block_filled

    @property
    def prop_partial_block(self):
        """Proportion filled in partial block in progress bar."""
        return self._n_col * self.prop_complete - self.n_block_filled

    @property
    def filled_blocks(self):
        """Filled blocks string."""
        return self.GLYPHS[-1] * self.n_block_filled

    @property
    def empty_blocks(self):
        """Empty blocks string."""
        if self.prop_partial_block == 0:
            return self.GLYPHS[0] * self.n_block_empty
        else:
            return self.GLYPHS[0] * (self.n_block_empty - 1)

    @property
    def partial_block(self):
        """Partial block character."""
        if self.prop_partial_block == 0:
            return ''
        else:
            return self.GLYPHS[int(len(self.GLYPHS) * self.prop_partial_block)]

    @property
    def progress_bar(self):
        """Progress bar string."""
        return f'|{self.filled_blocks}{self.partial_block}{self.empty_blocks}|'

    @property
    def bar_color(self):
        """CSS color property for HTML progress bar."""
        if self.counter == self.n_iter:
            return 'var(--jp-success-color1)'
        elif self._active:
            return 'var(--jp-brand-color1)'
        else:
            return 'var(--jp-error-color1)'

    @property
    def stats(self):
        """Comma-delimited string list of statistic key=value pairs."""
        return ', '.join(f'{k}={v:#.3g}' for k, v in self._stats_dict.items())

    @property
    def prefix(self):
        """Text to prefix progress bar with."""
        return (
            f'{self.description + ": "if self.description else ""}'
            f'{self.perc_complete}')

    @property
    def postfix(self):
        """Text to postfix progress bar with."""
        return (
            f'{self.counter}/{self.n_iter} '
            f'[{self.elapsed_time}<{self.est_remaining_time}, '
            f'{self.iter_rate}'
            f'{", " + self.stats if self._stats_dict else ""}]')

    def reset(self):
        """Reset progress bar state."""
        self._counter = 0
        self._active = True
        self._start_time = timer()
        self._stats_dict = {}

    def update(self, iter, iter_dict=None, refresh=True):
        """Update progress bar state

        Args:
            iter (int): New value for iteration counter.
            iter_dict (None or Dict[str, float]): Dictionary of iteration
                statistics key-value pairs to use to update postfix stats.
            refresh (bool): Whether to refresh display(s).
        """
        if iter == 0:
            self.reset()
        else:
            self.counter = iter
            if iter_dict is not None:
                _update_stats_running_means(iter, self._stats_dict, iter_dict)
            self._elapsed_time = timer() - self._start_time
        if refresh:
            self.refresh()

    def refresh(self):
        """Refresh visual display(s) of progress bar."""
        for display in self._displays:
            display.update(self)

    def __str__(self):
        return f'{self.prefix}{self.progress_bar}{self.postfix}'

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return f'''
        <div style="line-height: 28px; width: 100%; display: flex;
                    flex-flow: row wrap; align-items: center;
                    position: relative; margin: 2px;">
          <label style="margin-right: 8px; flex-shrink: 0;
                        font-size: var(--jp-code-font-size);
                        font-family: var(--jp-code-font-family);">
            {html.escape(self.prefix).replace(' ', '&nbsp;')}
          </label>
          <div role="progressbar" aria-valuenow="{self.prop_complete}"
               aria-valuemin="0" aria-valuemax="1"
               style="position: relative; flex-grow: 1; align-self: stretch;
                      margin-top: 4px; margin-bottom: 4px;  height: initial;
                      background-color: #eee;">
            <div style="background-color: {self.bar_color}; position: absolute;
                        bottom: 0; left: 0; width: {self.perc_complete};
                        height: 100%;"></div>
          </div>
          <div style="margin-left: 8px; flex-shrink: 0;
                      font-family: var(--jp-code-font-family);
                      font-size: var(--jp-code-font-size);">
            {html.escape(self.postfix)}
          </div>
        </div>
        '''

    def __enter__(self):
        self.reset()
        if self._displays is None:
            self._displays = [
                ipython_display(self, display_id=True)
                if _in_zmq_interactive_shell()
                else FileDisplay(self._position)]
        return self

    def __exit__(self, *args):
        self._active = False
        if self.counter != self.n_iter:
            self.refresh()
        return False


class FileDisplay:
    """Use file which supports ANSI escape sequences as an updatable display"""

    CURSOR_UP = '\x1b[A'
    """ANSI escape sequence to move cursor up one line."""

    CURSOR_DOWN = '\x1b[B'
    """ANSI escape sequence to move cursor down one line."""

    def __init__(self, position=(0, 1), file=None):
        """
        Args:
            position (Tuple[int, int]): Tuple specifying position of
                display line within a sequence lines with first entry
                corresponding to zero-indexed line and the second entry
                the total number of lines.
            file (None or File): File object to write updates to. Must support
                ANSI escape sequences `\\x1b[A}` (cursor up) and `\\x1b[B`
                (cursor down) for manipulating write position. Defaults to
                `sys.stdout` if `None`.
        """
        self._position = position
        self._file = file if file is not None else sys.stdout
        self._last_string_length = 0
        self._file.write('\n')
        self._file.flush()

    def _move_line(self, offset):
        self._file.write(self.CURSOR_DOWN * offset + self.CURSOR_UP * -offset)
        self._file.flush()

    def update(self, obj):
        self._move_line(self._position[0] - self._position[1])
        string = str(obj)
        self._file.write(f'{string: <{self._last_string_length}}\r')
        self._last_string_length = len(string)
        self._move_line(self._position[1] - self._position[0])
        self._file.flush()


class _ProxyProgressBar:
    """Proxy progress bar that outputs progress updates to a queue.

    Intended for communicating progress updates from a child to parent process
    when distributing tasks across multiple processes.
    """

    def __init__(self, n_iter, job_id, iter_queue):
        """
        Args:
            n_iter (int): Number of iterations to iterate over.
            job_id (int): Unique integer identifier for progress bar amongst
                other progress bars sharing same `iter_queue` object.
            iter_queue (Queue): Shared queue object that progress updates are
                pushed to.
        """
        self._n_iter = n_iter
        self._job_id = job_id
        self._iter_queue = iter_queue

    def __len__(self):
        return self._n_iter

    def __enter__(self):
        self._iter_queue.put((self._job_id, 0, None))
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for iter in range(self._n_iter):
            iter_dict = {}
            yield iter, iter_dict
            self._iter_queue.put((self._job_id, iter + 1, iter_dict))


if TQDM_AVAILABLE:

    class TqdmProgressBar(BaseProgressBar):
        """Wrapper of `tqdm` with same interface as `ProgressBar`."""

        def __init__(self, n_iter, description=None, position=(0, 1)):
            super().__init__(n_iter, description, position)
            self._stats_dict = {}
            self._tqdm_obj = None

        def update(self, iter, iter_dict=None, refresh=True):
            if self._tqdm_obj is None:
                raise RuntimeError(
                    'Must enter object first in context manager.')
            if iter == 0:
                self._tqdm_obj.reset()
            elif not self._tqdm_obj.disable:
                self._tqdm_obj.update(iter - self._tqdm_obj.n)
                if iter_dict is not None:
                    _update_stats_running_means(
                        iter, self._stats_dict, iter_dict)
                    self._tqdm_obj.set_postfix(self._stats_dict)
                if iter == self._n_iter:
                    self._tqdm_obj.close()
                if refresh:
                    self._tqdm_obj.refresh()

        def __enter__(self):
            self._tqdm_obj = tqdm.trange(
                self._n_iter, desc=self._description,
                position=self._position[0]).__enter__()
            return self

        def __exit__(self, *args):
            return self._tqdm_obj.__exit__(*args)
