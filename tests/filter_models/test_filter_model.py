import numpy as np
import pytest

from adaptive_filter.filter_models.filter_model import FilterModel


# creating a sample pytest model
@pytest.fixture
def sample_model():
    # creating a sample model
    filter_model = FilterModel(mu=0.1, filter_order=3)
    # setting the weights manually
    filter_model.W = np.array([1.0, -2.0, 0.5])
    return filter_model


# testing noise estimate function
def test_noise_estimate(sample_model):
    x_n = np.array([2.0, 3.0, 4.0])
    assert sample_model.noise_estimate(x_n) == pytest.approx(-2.0)


# testing the error function
def test_error(sample_model):
    d_n = 5.0
    noise_estimate = 3.5
    assert sample_model.error(d_n, noise_estimate) == pytest.approx(1.5)


# testing the update step
def test_update_step(sample_model):
    e_n = 5.0
    x_n = np.array([2.0, 3.0, 4.0])
    output = sample_model.update_step(e_n, x_n)
    assert isinstance(output, np.ndarray)
    assert output.shape == x_n.shape
    assert np.all(output == 0.0)


# testing the filter function
def test_filter():
    model = FilterModel(mu=0.1, filter_order=1)
    # overriding update step
    model.update_step = lambda e_n, x_n: np.array([0.0])

    d = np.linspace(1, 5, 6)
    x = np.linspace(0.5, 2.5, 5)
    clean = np.linspace(1, 5, 8)
    if d.shape[0] < x.shape[0]:
        x = x[: d.shape[0]]
    if x.shape[0] < d.shape[0]:
        d = d[: x.shape[0]]
    if d.shape[0] < clean.shape[0]:
        clean = clean[: d.shape[0]]
    if clean.shape[0] < d.shape[0]:
        d = d[: clean.shape[0]]
        x = x[: clean.shape[0]]
    # checking the signal shapes
    assert d.shape == x.shape
    assert d.shape == clean.shape

    # testing results when metrics not returned
    no_metrics_results = model.filter(d, x, clean)
    assert isinstance(no_metrics_results, tuple) and len(no_metrics_results) == 2

    # testing the result output when metrics returned
    result = model.filter(d, x, clean, return_metrics=True)
    assert isinstance(result, tuple) and len(result) == 8

    # checking the outputs
    (
        error,
        noise_estimate,
        adapt_mse,
        speech_mse,
        snr_res,
        delta_snr,
        elapsed_time,
        conv_time,
    ) = result
    assert isinstance(error, np.ndarray) and error.shape == (5,)
    assert isinstance(noise_estimate, np.ndarray) and error.shape == (5,)

    # checking metrics types
    for metric in (adapt_mse, speech_mse, snr_res, delta_snr, elapsed_time, conv_time):
        assert isinstance(metric, float)
