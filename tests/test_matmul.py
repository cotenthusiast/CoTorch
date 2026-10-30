
import math

import numpy as np
import pytest

import cotorch

ATOL = 1e-6
RTOL = 1e-5


def assert_matrices_equal(actual, expected):
    assert len(actual) == len(expected), "Row count mismatch"

    for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        assert len(actual_row) == len(expected_row), f"Column count mismatch in row {i}"

        for j, (a, e) in enumerate(zip(actual_row, expected_row)):
            assert not (math.isnan(a) or math.isnan(e)), f"NaN at ({i}, {j})"

            if math.isinf(a) or math.isinf(e):
                assert a == e, f"Infinity mismatch at ({i}, {j})"
            else:
                assert abs(a - e) <= ATOL + RTOL * abs(e), (
                    f"Mismatch at ({i}, {j}): actual={a}, expected={e}"
                )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 1, 1]] * 3,
            id="2x3-by-3x4",
            ),
        pytest.param(
            [[1, 1, 1]],
            [[1], [1], [1]],
            id="1x3-by-3x1",
        ),
        pytest.param(
            [[1], [1], [1]],
            [[1, 1]],
            id="3x1-by-1x2",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            id="ordinary",
        ),
        pytest.param(
            [[1.5, -2], [3, 4.25]],
            [[1, 0], [0, 1]],
            id="right-identity",
        ),
        pytest.param(
            [[1, 0], [0, 1]],
            [[1.5, -2], [3, 4.25]],
            id="left-identity",
        ),
        pytest.param(
            [[1.5, -2], [3, 4.25]],
            [[0, 0], [0, 0]],
            id="zero",
        ),
        pytest.param(
            [[0.1, 0.2]],
            [[0.3], [0.4]],
            id="fractional",
        ),
    ],
)
def test_matmul_matches_numpy(a, b):
    actual = cotorch.matmul(a, b)

    expected = (
            np.asarray(a, dtype=np.float32)
            @ np.asarray(b, dtype=np.float32)
    ).tolist()

    assert_matrices_equal(actual, expected)


def test_known_result():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[7, 8], [9, 10], [11, 12]]

    assert_matrices_equal(
        cotorch.matmul(a, b),
        [[58, 64], [139, 154]],
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2], [3, 4]],
            id="incompatible-dimensions",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
            id="same-incompatible-shapes",
        ),
        pytest.param([], [[1, 2], [3, 4]], id="empty-left"),
        pytest.param([[1, 2], [3, 4]], [], id="empty-right"),
        pytest.param([[]], [[1, 2], [3, 4]], id="no-columns-left"),
        pytest.param([[1, 2], [3, 4]], [[]], id="no-columns-right"),
        pytest.param([], [], id="both-empty"),
        pytest.param(
            [[1, 2], [3]],
            [[1, 2], [3, 4]],
            id="irregular-left",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            [[1, 2], [3]],
            id="irregular-right",
        ),
        pytest.param(
            [[1, 2], [3, 4], [5, 6, 7]],
            [[1, 2], [3, 4]],
            id="longer-row-left",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4], [5, 6, 7]],
            id="longer-row-right",
        ),
    ],
)
def test_invalid_inputs(a, b):
    with pytest.raises(ValueError):
        cotorch.matmul(a, b)
