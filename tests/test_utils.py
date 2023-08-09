import numpy as np
import pytest

import mici

SEED = 3046987125


@pytest.fixture()
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(params=(1, 3, (3, 2), (5, 5, 1), (10, 10, 10)))
def array_equal_pair(rng, request):
    array = rng.standard_normal(request.param)
    return array, array.copy()


def test_hash_array_equal(array_equal_pair):
    array_1, array_2 = array_equal_pair
    assert mici.utils.hash_array(array_1) == mici.utils.hash_array(array_2)


@pytest.fixture(params=((1, 1), (1, 3), ((5, 5), (5, 5)), ((5, 5, 1), (3, 2))))
def array_not_equal_pair(rng, request):
    shape_1, shape_2 = request.param
    return rng.standard_normal(shape_1), rng.standard_normal(shape_2)


def test_hash_array_not_equal(array_not_equal_pair):
    array_1, array_2 = array_not_equal_pair
    assert mici.utils.hash_array(array_1) != mici.utils.hash_array(array_2)


def get_val(obj):
    if isinstance(obj, mici.utils.LogRepFloat):
        return obj.val
    else:
        return obj


class TestLogRepFloat:
    VALS = sorted((0.0, 0.9, 1, 1.1, 2.1, 120.0))
    FIXED = VALS[1]

    @pytest.fixture(params=VALS)
    def val_pair(self, request):
        val = request.param
        return mici.utils.LogRepFloat(val), val

    @pytest.fixture(params=VALS)
    def other_pair(self, request):
        val = request.param
        return mici.utils.LogRepFloat(val), val

    def test_greater_than(self, val_pair, other_pair):
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] > other_pair[0]) == (val_pair[1] > other_pair[1])
            assert (val_pair[0] > other_pair[1]) == (val_pair[1] > other_pair[0])

    def test_less_than(self, val_pair, other_pair):
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] < other_pair[0]) == (val_pair[1] < other_pair[1])
            assert (val_pair[0] < other_pair[1]) == (val_pair[1] < other_pair[0])

    def test_greater_than_or_equal(self, val_pair, other_pair):
        assert val_pair[0] >= val_pair[0]
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] >= other_pair[0]) == (val_pair[1] >= other_pair[1])
            assert (val_pair[0] >= other_pair[1]) == (val_pair[1] >= other_pair[0])

    def test_less_than_or_equal(self, val_pair, other_pair):
        assert val_pair[0] <= val_pair[0]
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] <= other_pair[0]) == (val_pair[1] <= other_pair[1])
            assert (val_pair[0] <= other_pair[1]) == (val_pair[1] <= other_pair[0])

    def test_equal_to(self, val_pair, other_pair):
        assert val_pair[0] == val_pair[0]
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] == other_pair[0]) == (val_pair[1] == other_pair[1])
            assert (val_pair[0] == other_pair[1]) == (val_pair[1] == other_pair[0])

    def test_not_equal_to(self, val_pair, other_pair):
        assert val_pair[0] == val_pair[0]
        if val_pair[1] != other_pair[1]:
            assert (val_pair[0] != other_pair[0]) == (val_pair[1] != other_pair[1])
            assert (val_pair[0] != other_pair[1]) == (val_pair[1] != other_pair[0])

    def test_mult(self, val_pair, other_pair):
        assert np.isclose(
            get_val(val_pair[0] * other_pair[0]),
            get_val(val_pair[1] * other_pair[1]),
        )
        assert np.isclose(
            get_val(val_pair[0] * other_pair[1]),
            get_val(val_pair[1] * other_pair[0]),
        )

    def test_div(self, val_pair, other_pair):
        if other_pair[1] != 0:
            assert np.isclose(
                get_val(val_pair[0] / other_pair[0]),
                get_val(val_pair[1] / other_pair[1]),
            )
            assert np.isclose(
                get_val(val_pair[0] / other_pair[1]),
                get_val(val_pair[1] / other_pair[0]),
            )

    def test_add(self, val_pair, other_pair):
        assert np.isclose(
            get_val(val_pair[0] + other_pair[0]),
            get_val(val_pair[1] + other_pair[1]),
        )
        assert np.isclose(
            get_val(val_pair[0] + other_pair[1]),
            get_val(val_pair[1] + other_pair[0]),
        )

    def test_sub(self, val_pair, other_pair):
        assert np.isclose(
            get_val(val_pair[0] - other_pair[0]),
            get_val(val_pair[1] - other_pair[1]),
        )
        assert np.isclose(
            get_val(val_pair[0] - other_pair[1]),
            get_val(val_pair[1] - other_pair[0]),
        )

    def test_neg(self, val_pair):
        assert np.isclose(-val_pair[0], -val_pair[1])

    def test_iadd(self, val_pair, other_pair):
        sum_ = val_pair[0]
        sum_ += other_pair[0]
        assert np.isclose(get_val(sum_), val_pair[1] + other_pair[1])
        sum_ = other_pair[0]
        sum_ += val_pair[1]
        assert np.isclose(get_val(sum_), val_pair[1] + other_pair[1])

    def test_str(self, val_pair):
        assert isinstance(str(val_pair[0]), str)

    def test_repr(self, val_pair):
        assert isinstance(repr(val_pair[0]), str)

    def test_overflow(self):
        assert mici.utils.LogRepFloat(log_val=1e6) == np.inf

    def test_underflow(self):
        assert mici.utils.LogRepFloat(log_val=-1e6) == 0.0

    def test_neg_init(self):
        with pytest.raises(ValueError, match="non-negative"):
            mici.utils.LogRepFloat(val=-1)

    def test_no_init(self):
        with pytest.raises(ValueError, match="One of val or log_val must be specified"):
            mici.utils.LogRepFloat()

    def test_both_init(self):
        with pytest.raises(ValueError, match="Specify only one of val and log_val."):
            mici.utils.LogRepFloat(val=1.0, log_val=0.0)
