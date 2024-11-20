from dataclasses import dataclass

import pandas as pd

from autofix.ml.lib.data_schemas import EvaluationSchema
from autofix.ml.lib.metrics_impl import ExactMatchAtKAccuracy
from autofix.ml.lib.metrics_impl import FloatMetric
from autofix.ml.lib.metrics_impl import PassAtKAccuracy
from autofix.ml.lib.metrics_impl import SampleCount


@dataclass
class AutofixMetricAggregator:
    """This class aggregates the metrics once they are in EvaluationSchema

    """

    def aggregate_all_rules_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Takes the checked predictions, scores and computes various metrics based on them.

        Output `DataFrame` has a column `rule_id` and number of columns corresponding to metrics.
        """
        metrics: list[FloatMetric] = [
            PassAtKAccuracy(),
            ExactMatchAtKAccuracy(),
            SampleCount(),
        ]
        existing_rules = list(predictions[EvaluationSchema.rule].unique())

        rule_to_predictions: dict[str, Any] = {  # type: ignore
            rule_id: predictions[predictions[EvaluationSchema.rule] == rule_id]
            for rule_id in existing_rules
        }

        # Create an empty metrics dataframe, containing only `rule_id`s. Metric columns will be
        # appended on the right.
        result: pd.DataFrame = pd.DataFrame({"rule_id": existing_rules})
        for m in metrics:
            metric_results = result.apply(
                lambda row: m.compute_metric_for_predictions(
                    rule_to_predictions[row.rule_id]
                ),
                axis=1,
            )
            result = pd.concat([result, metric_results], axis=1)

        percentage_cols = result.select_dtypes(include="number").columns.difference(
            ["samples"]
        )
        result[percentage_cols] = result[percentage_cols].mul(100).round(2)
        return result
