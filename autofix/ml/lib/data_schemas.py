"""We use strict dataframe schema validations.

Use `pandera` for data validations. See below for examples.
"""

import numpy as np
import pandera as pa
from pandera.typing import Int32
from pandera.typing import Object
from pandera.typing import Series


class LabelledDataSchema(pa.SchemaModel):
    rule: Series[str]
    message: Series[str]
    line_number: Series[Int32]
    line_end: Series[Int32]
    col_begin: Series[Int32]
    col_end: Series[Int32]
    severity: Series[Int32]
    event_id: Series[Int32]
    pre_file: Series[str]
    post_file: Series[str]
    repo: Series[str]
    pre_filename: Series[str]
    post_filename: Series[str]
    pre_sha: Series[str]
    post_sha: Series[str]
    pre_reduced: Series[str]
    post_reduced: Series[str]
    reduction_line_map: Series[Object]
    reduction_match_line_num: Series[Int32]
    language: Series[str]
    jj: Series[str]

    @pa.check("reduction_line_map")
    def is_list_of_int32(cls, s: Series) -> Series[bool]:
        # `.pipe()` is needed to reassure checker
        return s.apply(
            lambda elem: isinstance(elem, (list, tuple, np.ndarray))
            and (len(elem) == 0 or isinstance(elem[0], (int, np.int32)))
        ).pipe(Series[bool])


class PredictionSchema(LabelledDataSchema):
    predictions: Series[Object]  # List[str], checked at runtime

    @pa.check("predictions")
    def is_list_of_strings(cls, s: Series) -> Series[bool]:
        # `.pipe()` is needed to reassure checker
        return s.apply(
            lambda elem: isinstance(elem, (list, tuple, np.ndarray))
            and (len(elem) == 0 or isinstance(elem[0], (str, np.str_)))
        ).pipe(Series[bool])


class EvaluationSchema(PredictionSchema):
    true_fix: Series[Object]  # List[bool], checked at runtime
    eval_status: Series[Object]  # List[str], checked at runtime
    exact_match: Series[Object]  # List[bool], checked at runtime

    @pa.check("predictions", "eval_status")
    def is_list_of_strings(cls, s: Series) -> Series[bool]:
        return super().is_list_of_strings(s)

    @pa.check("true_fix", "exact_match")
    def is_list_of_bool(cls, s: Series) -> Series[bool]:
        # `.pipe()` is needed to reassure checker
        return s.apply(
            lambda elem: isinstance(elem, (list, tuple, np.ndarray))
            and (len(elem) == 0 or isinstance(elem[0], (bool, np.bool_)))
        ).pipe(Series[bool])
