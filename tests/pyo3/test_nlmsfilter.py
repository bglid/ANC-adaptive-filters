import numpy as np
import pytest

from adaptif import NLMSFilter


@pytest.fixture
def filter():
    return NLMSFilter(
        mu=0.1,
        window_size=1,
    )


def test_window_size():
    filter = NLMSFilter(mu=1.0, window_size=1024)
    assert filter.window_size == 1024


def test_adapt(filter):
    # TODO: check that weights update

    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    output = filter.adapt(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape


def test_filter(filter):
    # TODO: check that weights don't update

    input_signal = np.linspace(1, 5, 5)
    noise_ref = np.linspace(0.5, 2.5, 5)

    output = filter.filter(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape


@pytest.mark.parametrize(
    "input, noise",
    [
        (np.linspace(1, 5, 5), np.array([])),
        (np.array([]), np.linspace(0.5, 2.5, 5)),
        (np.array([]), np.array([])),
    ],
)
def test_empty_input(filter, input, noise):
    with pytest.raises(ValueError):
        filter.adapt(input, noise)
    with pytest.raises(ValueError):
        filter.filter(input, noise)


@pytest.mark.parametrize("fn_name", ["adapt", "filter"])
def test_signal_lengths(filter, fn_name):
    filter_fn = getattr(filter, fn_name)

    short_input = np.linspace(1, 5, 4)
    long_input = np.linspace(1, 5, 6)
    noise_ref = np.linspace(0.5, 2.5, 5)

    short_output = filter_fn(short_input, noise_ref)
    assert isinstance(short_output, np.ndarray)
    assert short_output.shape == short_input.shape

    with pytest.raises(ValueError):
        filter_fn(long_input, noise_ref)


@pytest.mark.parametrize("fn_name", ["adapt", "filter"])
def test_input_contiguous(filter, fn_name):
    filter_fn = getattr(filter, fn_name)

    input_signal = np.linspace(1, 5, 10)
    noise_ref = np.linspace(0.5, 2.5, 10)

    # regular slices are contiguous -> allowed
    filter_fn(input_signal[:5], noise_ref[:5])

    # strided slices are not contiguous -> not allowed
    with pytest.raises(ValueError):
        filter_fn(input_signal[::2], noise_ref[:5])
    with pytest.raises(ValueError):
        filter_fn(input_signal[:5], noise_ref[::2])
    with pytest.raises(ValueError):
        filter_fn(input_signal[::2], noise_ref[::2])

    # calling np.ascontiguousarray makes it possible to use non-contiguous arrays as input
    filter_fn(
        np.ascontiguousarray(input_signal[::2]), np.ascontiguousarray(noise_ref[::2])
    )
