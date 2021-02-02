import pytest
import pmvae.silly_sum as silly_sum
import pandas as pd
import numpy as np
import os

from click.testing import CliRunner


def path_name():
    return os.path.join(os.path.dirname(__file__), 'data', 'numbers.txt')

@pytest.fixture
def fixed_number_array():
    """ Returns some numbers from a file """
    path = path_name()
    with open(path) as f:
        return np.array([float(l) for l in f])


expected_file_result = 47.8999


def test_silly_sum_simple():
    assert 4 == silly_sum.silly_sum(pd.Series([1, 2, 3]))


def test_silly_sum_raise_exception_on_illegal_step():
    with pytest.raises(ValueError):
        silly_sum.silly_sum([], -1)


def test_silly_sum_stepsize_one():
    # to demonstrate handling random data
    np.random.seed(4200)  # import to fix seed, otherwise hard to reproduce
    data = np.random.uniform(-1.0, 1.0, 10)
    assert pytest.approx(sum(data)) == silly_sum.silly_sum(data, 1)


def test_silly_sum_from_file():
    assert pytest.approx(expected_file_result) == silly_sum.silly_sum_from_file(
        path_name(), 2)


@pytest.mark.parametrize("step,expected", [
    (1, 49.9999),
    (2, expected_file_result),
    (3, 4.3),
    (4, 46.9),
    (5, 1.0),
    (6, 1.0)
])
def test_silly_sum_parametrized(fixed_number_array, step, expected):
    # demonstrating parametrized testing
    assert pytest.approx(expected) == silly_sum.silly_sum(fixed_number_array,
                                                          step)


def test_command_line_interface():
    """ Test the CLI. """
    runner = CliRunner()

    help_result = runner.invoke(silly_sum.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output


def test_end_to_end():
    # Integration test
    runner = CliRunner()
    result = runner.invoke(silly_sum.main, ['--step', '2', path_name()])

    assert pytest.approx(expected_file_result) == float(result.output)
    assert result.exit_code == 0
