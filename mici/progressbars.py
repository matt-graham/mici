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
    import google.colab
    ON_COLAB = True
except ImportError:
    ON_COLAB = False


def _in_zmq_interactive_shell():
    """Check if in interactive ZMQ shell which supports updateable displays"""
    if not IPYTHON_AVAILABLE:
        return False
    elif ON_COLAB:
        return True
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


def _create_display(obj, position):
    """Create an updateable display object.

    Args:
        obj (object): Initial object to display.
        position (Tuple[int, int]): Tuple specifying position of display within
            a sequence of displays with first entry corresponding to the
            zero-indexed position and the second entry the total number of
            displays.

    Returns:
        Object with `update` method to update displayed content.
    """
    if _in_zmq_interactive_shell():
        return ipython_display(obj, display_id=True)
    else:
        return FileDisplay(position)


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

    def __init__(self, sequence, description, position=(0, 1)):
        """
        Args:
            sequence (Sequence): Sequence to iterate over. Must be iterable AND
                have a defined length such that `len(sequence)` is valid.
            description (None or str): Description of task to prefix progress
                bar with.
            position (Tuple[int, int]): Tuple specifying position of progress
                bar within a sequence with first entry corresponding to
                zero-indexed position and the second entry the total number of
                progress bars.
        """
        self._sequence = sequence
        self._description = description
        self._position = position
        self._active = False
        self._n_iter = len(sequence)

    @property
    def sequence(self):
        """Sequence iterated over."""
        return self._sequence

    @sequence.setter
    def sequence(self, value):
        if self._active:
            raise RuntimeError('Cannot set sequence of active progress bar.')
        else:
            self._sequence = value
            self._n_iter = len(value)

    @property
    def n_iter(self):
        return self._n_iter

    def __iter__(self):
        for i, val in enumerate(self.sequence):
            iter_dict = {}
            yield val, iter_dict
            self.update(i + 1, iter_dict, refresh=True)

    def __len__(self):
        return self._n_iter

    @abc.abstractmethod
    def update(self, iter_count, iter_dict, refresh=True):
        """Update progress bar state.

        Args:
            iter_count (int): New value for iteration counter.
            iter_dict (None or Dict[str, float]): Dictionary of iteration
                statistics key-value pairs to use to update postfix stats.
            refresh (bool): Whether to refresh display(s).
        """

    def __enter__(self):
        """Set up progress bar and any associated resource."""
        self._active = True
        return self

    def __exit__(self, *args):
        """Close down progress bar and any associated resources."""
        self._active = False
        return False


class DummyProgressBar(BaseProgressBar):
    """Placeholder progress bar which does not display progress updates."""

    def update(self, iter_count, iter_dict, refresh=True):
        pass


class ProgressBar(BaseProgressBar):
    """Iterable object for tracking progress of an iterative task.

    Implements both string and HTML representations to allow richer
    display in interfaces which support HTML output, for example Jupyter
    notebooks or interactive terminals.
    """

    GLYPHS = ' ▏▎▍▌▋▊▉█'
    """Characters used to create string representation of progress bar."""

    def __init__(self, sequence, description=None, position=(0, 1),
                 displays=None, n_col=10, unit='it', min_refresh_time=0.25):
        """
        Args:
            sequence (Sequence): Sequence to iterate over. Must be iterable AND
                have a defined length such that `len(sequence)` is valid.
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
            min_referesh_time (float): Minimum time in seconds between each
                refresh of progress bar visual representation.
        """
        super().__init__(sequence, description, position)
        self._n_col = n_col
        self._unit = unit
        self._counter = 0
        self._start_time = None
        self._elapsed_time = 0
        self._stats_dict = {}
        self._displays = displays
        self._min_refresh_time = min_refresh_time

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
            return 'var(--jp-success-color1, #4caf50)'
        elif self._active:
            return 'var(--jp-brand-color1, #2196f3)'
        else:
            return 'var(--jp-error-color1, #f44336)'

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
        self._start_time = timer()
        self._last_refresh_time = -float('inf')
        self._stats_dict = {}

    def update(self, iter_count, iter_dict=None, refresh=True):
        """Update progress bar state

        Args:
            iter_count (int): New value for iteration counter.
            iter_dict (None or Dict[str, float]): Dictionary of iteration
                statistics key-value pairs to use to update postfix stats.
            refresh (bool): Whether to refresh display(s).
        """
        if iter_count == 0:
            self.reset()
        else:
            self.counter = iter_count
            if iter_dict is not None:
                _update_stats_running_means(
                    iter_count, self._stats_dict, iter_dict)
            self._elapsed_time = timer() - self._start_time
        if refresh and iter_count == self.n_iter or (
                timer() - self._last_refresh_time > self._min_refresh_time):
            self.refresh()
            self._last_refresh_time = timer()

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
                        font-size: var(--jp-code-font-size, 13px);
                        font-family: var(--jp-code-font-family, monospace);">
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
                      font-family: var(--jp-code-font-family, monospace);
                      font-size: var(--jp-code-font-size, 13px);">
            {html.escape(self.postfix)}
          </div>
        </div>
        '''

    def __enter__(self):
        super().__enter__()
        self.reset()
        if self._displays is None:
            self._displays = [_create_display(self, self._position)]
        return self

    def __exit__(self, *args):
        ret_val = super().__exit__()
        if self.counter != self.n_iter:
            self.refresh()
        return ret_val


class LabelledSequenceProgressBar(BaseProgressBar):
    """Iterable object for tracking progress of a sequence of labelled tasks."""

    def __init__(self, labelled_sequence, description=None, position=(0, 1),
                 displays=None):
        """
        Args:
            labelled_sequence (OrderedDict[str, Any]): Ordered dictionary with
                string keys corresponding to labels for stages represented by
                sequence and values the entries in the sequence being iterated
                over.
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
        """
        super().__init__(
            list(labelled_sequence.values()), description, position)
        self._labels = list(labelled_sequence.keys())
        self._description = description
        self._position = position
        self._counter = 0
        self._prev_time = None
        self._iter_times = [None] * self.n_iter
        self._stats_dict = {}
        self._displays = displays

    @property
    def counter(self):
        """Progress iteration count."""
        return self._counter

    @counter.setter
    def counter(self, value):
        self._counter = max(0, min(value, self.n_iter))

    @property
    def description(self):
        """Description of task being tracked."""
        return self._description

    @property
    def stats(self):
        """Comma-delimited string list of statistic key=value pairs."""
        return ', '.join(f'{k}={v:#.3g}' for k, v in self._stats_dict.items())

    @property
    def prefix(self):
        """Text to prefix progress bar with."""
        return f'{self.description + ": " if self.description else ""}'

    @property
    def postfix(self):
        """Text to postfix progress bar with."""
        return f' [{self.stats}]' if self._stats_dict else ''

    @property
    def completed_labels(self):
        """Labels corresponding to completed iterations."""
        return [
            f'{label} [{_format_time(time)}]' for label, time in
            zip(self._labels[:self._counter], self._iter_times[:self._counter])]

    @property
    def current_label(self):
        """Label corresponding to current iteration."""
        return self._labels[self._counter] if self.counter < self.n_iter else ''

    @property
    def unstarted_labels(self):
        """Labels corresponding to unstarted iterations."""
        return self._labels[self._counter + 1:]

    @property
    def progress_bar(self):
        """Progress bar string."""
        labels = self.completed_labels
        if self.counter < self.n_iter:
            labels.append(self.current_label)
        return ' > '.join(labels)

    def reset(self):
        """Reset progress bar state."""
        self._counter = 0
        self._prev_time = timer()
        self._iter_times = [None] * self.n_iter
        self._stats_dict = {}

    def update(self, iter_count, iter_dict=None, refresh=True):
        """Update progress bar state

        Args:
            status (string): New value for status string.
            iter_dict (None or Dict[str, float]): Dictionary of iteration
                statistics key-value pairs to use to update postfix stats.
            refresh (bool): Whether to refresh display(s).
        """
        if iter_count == 0:
            self.reset()
        else:
            self.counter = iter_count
            if iter_dict is not None:
                _update_stats_running_means(iter, self._stats_dict, iter_dict)
            curr_time = timer()
            self._iter_times[iter_count - 1] = curr_time - self._prev_time
            self._prev_time = curr_time
        if refresh:
            self.refresh()

    def refresh(self):
        """Refresh visual display(s) of status bar."""
        for display in self._displays:
            display.update(self)

    def __str__(self):
        return f'{self.prefix}{self.progress_bar}{self.postfix}'

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        html_string = f'''
        <div style="line-height: 24px; width: 100%; display: flex;
                    flex-flow: row wrap; align-items: center;
                    position: relative; margin: 2px;">
          <label style="flex-shrink: 0;
                        font-size: var(--jp-code-font-size, 13px);
                        font-family: var(--jp-code-font-family, monospace);">
            {html.escape(self.prefix).replace(' ', '&nbsp;')}
          </label>
        '''
        template_string = '''
          <div style="position: relative; flex-grow: 1; align-self: stretch;
                      margin: 1px; padding: 0px; text-align: center;
                      height: initial; background-color: {background_color};
                      color: {foreground_color}; border-radius: 5px;
                      border: 1px solid {foreground_color}; font-size: 90%;">
            {label}
          </div>
        '''
        for label in self.completed_labels:
            html_string += template_string.format(
                label=label, foreground_color='white',
                background_color='#4caf50')
        if self.counter != self.n_iter:
            html_string += template_string.format(
                label=self.current_label, foreground_color='white',
                background_color='#2196f3' if self._active else '#f44336')
        for label in self.unstarted_labels:
            html_string += template_string.format(
                label=label, foreground_color='#aaa', background_color='white')
        if self.postfix != '':
            html_string += f'''
              <div style="margin-left: 8px; flex-shrink: 0;
                          font-family: var(--jp-code-font-family, monospace);
                          font-size: var(--jp-code-font-size, 13px);">
                {html.escape(self.postfix)}
              </div>
            '''
        html_string += '</div>'
        return html_string

    def __enter__(self):
        super().__enter__()
        self.reset()
        if self._displays is None:
            self._displays = [_create_display(self, self._position)]
        self.refresh()
        return self

    def __exit__(self, *args):
        ret_val = super().__exit__()
        if self.counter != self.n_iter:
            self.refresh()
        return ret_val


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
        if self._position[0] == 0:
            self._file.write('\n' * self._position[1])
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

    def __init__(self, sequence, job_id, iter_queue):
        """
        Args:
            sequence (Sequence): Sequence to iterate over. Must be iterable AND
                have a defined length such that `len(sequence)` is valid.
            job_id (int): Unique integer identifier for progress bar amongst
                other progress bars sharing same `iter_queue` object.
            iter_queue (Queue): Shared queue object that progress updates are
                pushed to.
        """
        self._sequence = sequence
        self._n_iter = len(sequence)
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
        for i, val in enumerate(self._sequence):
            iter_dict = {}
            yield val, iter_dict
            self._iter_queue.put((self._job_id, i + 1, iter_dict))
