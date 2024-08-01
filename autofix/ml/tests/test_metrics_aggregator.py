import pandas as pd

from autofix.ml.lib.metrics_aggregator import AutofixMetricAggregator


def test_aggregate_all_rules_metrics(
    pass_at_k_data,
) -> None:
    metrics_aggregator = AutofixMetricAggregator()
    results: pd.DataFrame = metrics_aggregator.aggregate_all_rules_metrics(
        pass_at_k_data
    )
    assert set(results.columns) == set(
        [
            "rule_id",
            "exact@1",
            "exact@3",
            "exact@5",
            "pass@1",
            "pass@3",
            "pass@5",
            "samples",
        ]
    )
    assert set(results.rule_id) == set(["rule_a", "rule_b", "rule_c", "rule_d"])


def test_aggregate_all_rules_metrics_per_repo(
    pass_at_k_data,
) -> None:
    metrics_aggregator = AutofixMetricAggregator()
    results: pd.DataFrame = metrics_aggregator.aggregate_all_rules_metrics_per_repo(
        pass_at_k_data
    )
    assert set(results.columns) == set(
        [
            "rule_id",
            "repo_id",
            "exact@1",
            "exact@3",
            "exact@5",
            "pass@1",
            "pass@3",
            "pass@5",
            "samples",
        ]
    )
    assert set(results.rule_id) == set(["rule_a", "rule_b", "rule_c", "rule_d"])
    assert sorted(list(results.repo_id)) == ["repo_1", "repo_1", "repo_2", "repo_2"]
