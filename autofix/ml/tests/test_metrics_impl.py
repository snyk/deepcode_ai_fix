import pandas as pd
import pandas.testing as tpd

from autofix.ml.lib.metrics_impl import ExactMatchAtKAccuracy
from autofix.ml.lib.metrics_impl import FloatMetric
from autofix.ml.lib.metrics_impl import PassAtKAccuracy


def test_pass_at_k_accuracy(pass_at_k_data) -> None:
    m: FloatMetric = PassAtKAccuracy()
    assert m.get_string_name() == "pass@k accuracy"
    result = m.compute_metric_for_predictions(df=pass_at_k_data)

    expected_result = pd.Series()
    expected_result["pass@1"] = 1 / 4
    expected_result["pass@3"] = 3 / 4
    expected_result["pass@5"] = 4 / 4

    tpd.assert_series_equal(result, expected_result)


def test_pass_at_k_exact_match(pass_at_k_data) -> None:
    m: FloatMetric = ExactMatchAtKAccuracy()
    assert m.get_string_name() == "exact@k accuracy"
    result = m.compute_metric_for_predictions(df=pass_at_k_data)

    expected_result = pd.Series()
    expected_result["exact@1"] = 0 / 4
    expected_result["exact@3"] = 2 / 4
    expected_result["exact@5"] = 4 / 4

    tpd.assert_series_equal(result, expected_result)
