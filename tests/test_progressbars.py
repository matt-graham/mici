from collections import OrderedDict
from queue import SimpleQueue
from unittest import mock

import pytest

import mici


def test_format_time():
    assert mici.progressbars._format_time(100) == "01:40"
    assert mici.progressbars._format_time(10000) == "2:46:40"


def test_stats_running_means():
    means = {}
    vals = {"a": 1, "b": 2}
    mici.progressbars._update_stats_running_means(1, means, vals)
    assert means == vals
    mici.progressbars._update_stats_running_means(2, means, vals)
    assert means == vals


@pytest.mark.parametrize("ipython_available", [True, False])
@pytest.mark.parametrize("in_ipython_shell", [True, False])
def test_in_interactive_shell(monkeypatch, ipython_available, in_ipython_shell):
    monkeypatch.setattr("mici.progressbars.IPYTHON_AVAILABLE", ipython_available)
    if in_ipython_shell:
        ipython_mock = mock.Mock()
        ipython_mock.__module__ = "ipykernel.zmqshell"
        ipython_mock.__class__.__name__ = "ZMQInteractiveShell"
    else:
        ipython_mock = None
    monkeypatch.setattr("mici.progressbars.get_ipython", lambda: ipython_mock)
    assert mici.progressbars._in_interactive_shell() == (
        ipython_available and in_ipython_shell
    )


@pytest.fixture(
    params=(
        "DummyProgressBar",
        "SequenceProgressBar",
        "LabelledSequenceProgressBar",
        "_ProxySequenceProgressBar",
    ),
)
def progress_bar_and_sequence(request):
    if request.param == "LabelledSequenceProgressBar":
        sequence = OrderedDict()
        sequence["a"] = 1
        sequence["b"] = 2
        sequence["c"] = 3
        return (
            mici.progressbars.LabelledSequenceProgressBar(sequence),
            sequence.values(),
        )
    sequence = range(100)
    if request.param == "DummyProgressBar":
        return mici.progressbars.DummyProgressBar(sequence, None), sequence
    if request.param == "SequenceProgressBar":
        return mici.progressbars.SequenceProgressBar(sequence, None), sequence
    if request.param == "_ProxySequenceProgressBar":
        return (
            mici.progressbars._ProxySequenceProgressBar(sequence, 0, SimpleQueue()),
            sequence,
        )
    return None


def test_progress_bar_len(progress_bar_and_sequence):
    progress_bar, sequence = progress_bar_and_sequence
    assert len(progress_bar) == len(sequence)


def test_progress_bar_iter(progress_bar_and_sequence):
    progress_bar, sequence = progress_bar_and_sequence
    with progress_bar:
        for (val, iter_dict), val_orig in zip(progress_bar, sequence, strict=True):
            assert val == val_orig
            assert isinstance(iter_dict, dict)
