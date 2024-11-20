"""Metrics for the metric aggregator."""

from abc import ABCMeta
from abc import abstractmethod
from typing import List, Union, final

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from autofix.ml.lib.data_schemas import EvaluationSchema


class Metric(metaclass=ABCMeta):
    """Generic metric class."""

    @abstractmethod
    def compute_metric_for_predictions(
        self, df: DataFrame[EvaluationSchema]
    ) -> pd.Series:
        """Takes evaluations and computes a metric.

        Args:
            df: `DataFrame` with the schema EvaluationSchema
        """
        pass

    @abstractmethod
    def get_string_name(self) -> Union[str, List[str]]:
        pass


class FloatMetric(Metric):
    @abstractmethod
    def compute_metric_for_predictions(
        self, df: DataFrame[EvaluationSchema]
    ) -> pd.Series:
        pass


@final
class PassAtKAccuracy(FloatMetric):
    """Average % of datapoints that have at least one "pass" in `k` predictions (`pass@k`)."""

    step_size = 2

    def get_string_name(self) -> str:
        return "pass@k accuracy"

    @pa.check_types
    def compute_metric_for_predictions(
        self, df: DataFrame[EvaluationSchema]
    ) -> pd.Series:
        # `row["true_fix"]` is `List[bool]`` of length `k` containing if the `i`-th prediction has
        # fixed the problem.
        res = pd.Series(dtype="float64")

        assert df[EvaluationSchema.predictions] is not None
        num_predictions = len(df[EvaluationSchema.predictions].values[0])  # type: ignore

        for i in range(1, num_predictions, self.step_size):
            res[f"pass@{i}"] = np.float64(
                df.apply(lambda row: any(row["true_fix"][:i]), axis=1).sum() / len(df)
            )
        res[f"pass@{num_predictions}"] = np.float64(
            df.apply(lambda row: any(row["true_fix"][:num_predictions]), axis=1).sum()
            / len(df)
        )

        return res


@final
class ExactMatchAtKAccuracy(FloatMetric):
    """Evaluates exact match accuracy for k predictions."""

    step_size = 2

    def get_string_name(self) -> str:
        return "exact@k accuracy"

    @pa.check_types
    def compute_metric_for_predictions(
        self, df: DataFrame[EvaluationSchema]
    ) -> pd.Series:
        res = pd.Series(dtype="float64")

        assert df[EvaluationSchema.predictions] is not None
        num_predictions = len(df[EvaluationSchema.predictions].values[0])  # type: ignore

        for i in range(1, num_predictions, self.step_size):
            res[f"exact@{i}"] = np.float64(
                df.apply(lambda row: any(row["exact_match"][:i]), axis=1).sum()
                / len(df)
            )
        res[f"exact@{num_predictions}"] = np.float64(
            df.apply(
                lambda row: any(row["exact_match"][:num_predictions]), axis=1
            ).sum()
            / len(df)
        )

        return res


@final
class SampleCount(FloatMetric):
    """Counts the number of predictions."""

    def get_string_name(self) -> str:
        return "samples"

    @pa.check_types
    def compute_metric_for_predictions(
        self, df: DataFrame[EvaluationSchema]
    ) -> pd.Series:
        res = pd.Series(dtype="float64")

        assert df[EvaluationSchema.predictions] is not None
        res[self.get_string_name()] = len(df)
        return res
