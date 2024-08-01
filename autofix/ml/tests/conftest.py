import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from autofix.ml.lib.data_schemas import EvaluationSchema
from autofix.ml.lib.data_schemas import LabelledDataSchema


@pytest.fixture(autouse=True)
def mock_wandb_offline():
    with mock.patch.dict(os.environ, {"WANDB_DISABLED": "true"}):
        yield


df_dict = {
    LabelledDataSchema.rule: [
        "ArrayConstructor",
        "ArrayConstructor",
        "ArrayConstructor",
        "HttpToHttps",
        "ReactSetInnerHtml",
        "ReactSetInnerHtml",
        "ArrayConstructor",
    ],
    LabelledDataSchema.message: [
        "message_array_constructor",
        "message_array_constructor",
        "message_array_constructor",
        "message_http_to_https",
        "message_react_set_innet_html",
        "message_react_set_innet_html",
        "message_array_constructor",
    ],
    LabelledDataSchema.line_number: [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    LabelledDataSchema.line_end: [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    LabelledDataSchema.col_begin: [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    LabelledDataSchema.col_end: [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    LabelledDataSchema.severity: [
        1,
        0,
        0,
        0,
        1,
        4,
        8,
    ],
    LabelledDataSchema.event_id: [
        1,
        0,
        0,
        0,
        1,
        4,
        8,
    ],
    LabelledDataSchema.pre_file: [
        "//Full file var x = new Array();",
        "//Full file var x = new Array();",
        "//Full file var x = new Array();",
        "//Full file var http = require('http');",
        "//Full file page.innerHtml = myHtml();",
        "//Full file page.innerHtml = myHtml();",
        "//Full file var berkay = Array();",
    ],
    LabelledDataSchema.post_file: [
        "//Full file var x = [];",
        "//Full file var x = [];",
        "//Full file var y = [];",
        "//Full file var https = require('https');",
        "//Full file page.dangerouslySetInnerHtml() = myHtml();",
        "//Full file setInnerHtml(getRenderedHtml());",
        "//Full file var brot = createArray();",
    ],
    LabelledDataSchema.repo: [
        "repo1",
        "repo2",
        "repo3",
        "repo4",
        "repo5",
        "repo6",
        "repo7",
    ],
    LabelledDataSchema.pre_filename: [
        "foo.js",
        "bar.js",
        "foobar.js",
        "biz.js",
        "react.js",
        "html.js",
        "somefile.js",
    ],
    LabelledDataSchema.post_filename: [
        "foo.js",
        "bar.js",
        "foobar.js",
        "changed.js",
        "react.js",
        "html.js",
        "somepostfile.js",
    ],
    LabelledDataSchema.pre_sha: [
        "123456",
        "234567",
        "3456789",
        "4567891",
        "sdfsadg",
        "ssggttd",
        "mysha12",
    ],
    LabelledDataSchema.post_sha: [
        "saafad",
        "baasad",
        "caasad",
        "daasad",
        "sabnyt",
        "sghtsh",
        "yoursha12",
    ],
    LabelledDataSchema.pre_reduced: [
        "var x = new Array();",
        "var x = new Array();",
        "var x = new Array();",
        "var http = require('http');",
        "page.innerHtml = myHtml();",
        "page.innerHtml = myHtml();",
        "var berkay = Array();",
    ],
    LabelledDataSchema.post_reduced: [
        "var x = [];",
        "var x = [];",
        "var y = [];",
        "var https = require('https');",
        "page.dangerouslySetInnerHtml() = myHtml();",
        "setInnerHtml(getRenderedHtml());",
        "var brot = createArray();",
    ],
    LabelledDataSchema.reduction_line_map: [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ],
    LabelledDataSchema.reduction_match_line_num: [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    LabelledDataSchema.language: [
        "javascript",
        "javascript",
        "javascript",
        "javascript",
        "javascript",
        "javascript",
        "javascript",
    ],
    LabelledDataSchema.jj: [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
}


@pytest.fixture
def autofix_df() -> DataFrame[LabelledDataSchema]:
    df = pd.DataFrame(df_dict)
    int32_columns = [
        LabelledDataSchema.line_number,
        LabelledDataSchema.line_end,
        LabelledDataSchema.col_begin,
        LabelledDataSchema.col_end,
        LabelledDataSchema.severity,
        LabelledDataSchema.event_id,
        LabelledDataSchema.reduction_match_line_num,
    ]
    for col in int32_columns:
        df[col] = df[col].astype("int32")

    return DataFrame[LabelledDataSchema](df)


@pytest.fixture
def pass_at_k_data() -> DataFrame[EvaluationSchema]:
    """Fixture that returns simple predictions file for metrics aggregator to consume."""
    data = {
        EvaluationSchema.rule: ["rule_a", "rule_b", "rule_c", "rule_d"],
        EvaluationSchema.message: ["", "", "", ""],
        EvaluationSchema.line_number: [0, 0, 0, 0],
        EvaluationSchema.line_end: [0, 0, 0, 0],
        EvaluationSchema.col_begin: [0, 0, 0, 0],
        EvaluationSchema.col_end: [0, 0, 0, 0],
        EvaluationSchema.severity: [0, 0, 0, 0],
        EvaluationSchema.event_id: [0, 0, 0, 0],
        EvaluationSchema.pre_file: ["", "", "", ""],
        EvaluationSchema.post_file: ["", "", "", ""],
        EvaluationSchema.repo: ["repo_1", "repo_1", "repo_2", "repo_2"],
        EvaluationSchema.pre_filename: ["", "", "", ""],
        EvaluationSchema.post_filename: ["", "", "", ""],
        EvaluationSchema.pre_sha: ["", "", "", ""],
        EvaluationSchema.post_sha: ["", "", "", ""],
        EvaluationSchema.pre_reduced: ["", "", "", ""],
        EvaluationSchema.post_reduced: [
            "postcode1",
            "postcode2",
            "postcode3",
            "postcode4",
        ],
        EvaluationSchema.reduction_line_map: [[1], [1], [1], [1]],
        EvaluationSchema.reduction_match_line_num: [0, 0, 0, 0],
        EvaluationSchema.language: [
            "javascript",
            "javascript",
            "javascript",
            "javascript",
        ],
        EvaluationSchema.jj: [
            "",
            "",
            "",
            "",
        ],
        EvaluationSchema.predictions: [
            ["no exact match", "", "postcode1", "", "postcode1"],
            ["no exact match", "", "no exact match", "", "postcode2"],
            ["no exact match", "", "postcode3", "", "postcode3"],
            ["no exact match", "", "no exact match", "", "postcode4"],
        ],
        EvaluationSchema.true_fix: [
            [False, False, True, False, False],
            [False, False, False, False, True],
            [True, True, True, True, True],
            [False, True, True, False, False],
        ],
        EvaluationSchema.eval_status: [
            ["", "", "", "", ""],
            ["", "", "", "", ""],
            ["", "", "", "", ""],
            ["", "", "", "", ""],
        ],
        EvaluationSchema.exact_match: [
            [False, False, False, True, False],
            [False, False, False, False, True],
            [False, True, True, True, True],
            [False, True, True, False, False],
        ],
    }
    df = pd.DataFrame(data)
    int32_columns = [
        EvaluationSchema.line_number,
        EvaluationSchema.line_end,
        EvaluationSchema.col_begin,
        EvaluationSchema.col_end,
        EvaluationSchema.severity,
        EvaluationSchema.event_id,
        EvaluationSchema.reduction_match_line_num,
    ]
    for col in int32_columns:
        df[col] = df[col].astype("int32")

    df[EvaluationSchema.reduction_line_map] = df[
        EvaluationSchema.reduction_line_map
    ].apply(lambda row: [np.int32(el) for el in row])
    return DataFrame[EvaluationSchema](df)
