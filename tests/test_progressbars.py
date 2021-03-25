import pytest
import mici
from collections import OrderedDict
from queue import SimpleQueue


def test_format_time():
    assert mici.progressbars._format_time(100) == "01:40"
    assert mici.progressbars._format_time(10000) == "2:46:40"


def test_stats_runnin_means():
    means = {}
    vals = {"a": 1, "b": 2}
    mici.progressbars._update_stats_running_means(1, means, vals)
    assert means == vals
    mici.progressbars._update_stats_running_means(2, means, vals)
    assert means == vals


@pytest.fixture(
    params=(
        "DummyProgressBar",
        "SequenceProgressBar",
        "LabelledSequenceProgressBar",
        "_ProxySequenceProgressBar",
    )
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
    else:
        sequence = range(100)
        if request.param == "DummyProgressBar":
            return mici.progressbars.DummyProgressBar(sequence, None), sequence
        elif request.param == "SequenceProgressBar":
            return mici.progressbars.SequenceProgressBar(sequence, None), sequence
        elif request.param == "_ProxySequenceProgressBar":
            return (
                mici.progressbars._ProxySequenceProgressBar(sequence, 0, SimpleQueue()),
                sequence,
            )


def test_progress_bar_len(progress_bar_and_sequence):
    progress_bar, sequence = progress_bar_and_sequence
    assert len(progress_bar) == len(sequence)


def test_progress_bar_iter(progress_bar_and_sequence):
    progress_bar, sequence = progress_bar_and_sequence
    with progress_bar:
        for (val, iter_dict), val_orig in zip(progress_bar, sequence):
            assert val == val_orig
            assert isinstance(iter_dict, dict)

