import pytest

from adaptif import LMSFilter, NLMSFilter


@pytest.mark.parametrize("filter_algo", [LMSFilter, NLMSFilter])
def test_filter_bindings(filter_algo):
    filter = filter_algo(1.0, 1024)
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
    assert not hasattr(filter, "adapt_filter_impl")
