from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import autofix.ml.lib.cloud_manager as cm
import ml_infra.wandb.utils as wu


def plot_scores_per_types(
    metrics_per_rule: pd.DataFrame, plot_name: str, per_repo: bool = False
):
    rule_to_cat = cm.get_rule_categories()
    col_to_agg = {
        col: "mean"
        for col in metrics_per_rule.columns
        if col not in ["rule_id", "repo_id", "samples"]
    }
    col_to_agg["samples"] = "sum"
    metrics_per_rule["type"] = metrics_per_rule["rule_id"].map(
        lambda rule: rule_to_cat.get(rule, rule)
    )
    group_by_cols = ["repo_id", "type"] if per_repo else "type"
    metrics_per_type = metrics_per_rule.groupby(group_by_cols, as_index=False).agg(
        col_to_agg
    )
    wu.log_table(plot_name, metrics_per_type)


def plot_test_leakage(train_df, test_df):
    git_test_df = test_df.repo.str.split("/", expand=True).rename(
        {0: "org", 1: "repo"}, axis=1
    )
    git_train_df = train_df.repo.str.split("/", expand=True).rename(
        {0: "org", 1: "repo"}, axis=1
    )

    leaked_orgs = git_test_df.repo.isin(git_train_df.repo)
    leaked_repos = git_test_df.repo.isin(git_train_df.repo)
    leaked_files = test_df.pre_filename.isin(train_df.pre_filename)

    leaked_both = leaked_orgs & leaked_repos
    leaked_all = leaked_orgs & leaked_repos & leaked_files

    table_data = [
        [
            "#Total samples",
            "#Samples with leaked org",
            "#Samples with leaked orgs&repo",
            "#Samples with leaked org&repo&files",
        ],
        [len(test_df), leaked_orgs.sum(), leaked_both.sum(), leaked_all.sum()],
    ]
    fig = ff.create_table(table_data, height_constant=60)
    return fig


def plot_data_leakage(train_df, test_df):
    def extract_unique_orgs_and_repos(df):
        df = df.repo.str.split("/", expand=True).rename({0: "org", 1: "repo"}, axis=1)
        unique_orgs = set(df.org.tolist())
        unique_repos = set(df.repo.tolist())
        return unique_orgs, unique_repos

    train_orgs, train_repos = extract_unique_orgs_and_repos(train_df)
    test_orgs, test_repos = extract_unique_orgs_and_repos(test_df)
    train_org_repos = set(train_df.repo.tolist())
    test_org_repos = set(test_df.repo.tolist())
    table_data = [
        ["Counter", "Train", "Test", "Both"],
        [
            "Unique Organizations",
            len(train_orgs),
            len(test_orgs),
            len(train_orgs.intersection(test_orgs)),
        ],
        [
            "Unique Repositories",
            len(train_repos),
            len(test_repos),
            len(train_repos.intersection(test_repos)),
        ],
        [
            "Unique Organization/Repo",
            len(train_org_repos),
            len(test_org_repos),
            len(train_org_repos.intersection(test_org_repos)),
        ],
    ]
    fig = ff.create_table(table_data, height_constant=60)
    return fig


def plot_samples_per_rule(df):
    samples_per_rule = df.groupby("rule").size().sort_values(ascending=False)
    fig = px.bar(
        samples_per_rule,
        title="Number of samples by rule",
        color=samples_per_rule.values,
        color_continuous_scale="rdylgn",
        labels={"rule": "Rule Key", "value": "#Samples"},
    )
    fig.update_layout(title_font_size=16, font_size=10)
    return fig


def plot_samples_per_language(df):
    samples_per_rule = df.groupby("language").size().sort_values(ascending=False)
    fig = px.bar(
        samples_per_rule,
        title="Number of samples by language",
        labels={"language": "Language", "value": "#Samples"},
    )
    fig.update_layout(title_font_size=16, font_size=10)
    return fig


def log_exact_and_pass_metrics_table(df: pd.DataFrame, plot_name: str):
    fig = go.Figure(
        data=[
            go.Table(
                # rule_id requires a wider column, the rest is equally wide
                columnwidth=[500] + [200] * (len(df.columns) - 1),
                header=dict(
                    values=df.columns.to_list(),
                    align="center",
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="center",
                    height=30,
                ),
            )
        ]
    )
    wu.log_plot(plot_name, fig)


def log_error_rates(predictions: pd.DataFrame, plot_name: str) -> None:
    known_issues = {
        "Unexpected token",
        "has already been declared",
        "Missing semicolon",
        "Unexpected keyword",
        "Unterminated comment",
        "Unterminated string constant",
        "Unsyntactic break",
        "Unterminated template",
        "Unsyntactic continue",
        "Missing catch or finally clause",
        "Unexpected reserved word",
        "Unterminated regular expression",
        "'with' in strict mode",
        "Did not expect a type annotation here",
        "Legacy octal literals are not allowed in strict mode",
        "Adjacent JSX elements must be wrapped in an enclosing tag",
        "Unterminated JSX contents",
        "Expected corresponding JSX closing tag",
        "Argument name clash",
        "In strict mode code, functions can only be declared at top level or inside a block",
        "Missing initializer in destructuring declaration",
        "Invalid left-hand side in assignment expression",
        "Identifier directly after number",
        "Deleting local variable in strict mode",
        "Missing initializer in const declaration",
        "Binding 'arguments' in strict mode",
        "A class name is required",
        "Assigning to 'arguments' in strict mode",
        "Binding 'arguments' in strict mode",
        "A class name is required",
        "The only valid numeric escape in strict mode is",
        "Invalid regular expression flag",
        "'super' can only be used with function calls",
        "Expecting Unicode escape sequence",
        "Binding 'eval' in strict mode",
        "Invalid shorthand property initializer",
        "Leading decorators must be attached to a class declaration",
        "Private names are only allowed in property accesses",
        "JSX value should be either an expression or a quoted JSX text",
        "'interface' declarations must be followed by an identifier",
        "The only valid meta property for import is import.meta",
        "Binding invalid left-hand side in array destructuring pattern",
        "Unexpected character",
        "Binding member expression",
        "Unexpected digit after hash token",
        "`new.target` can only be used in functions or class properties",
        "ES2015 named imports do not destructure. Use another statement for destructuring after the import.",
        "'arguments' is only allowed in functions and class methods",
        "This experimental syntax requires enabling the parser plugin",
        "Empty Tree",
    }
    col_eval_status = "eval_status"
    col_counts = "counts"

    error_to_count: dict[str, int] = defaultdict(int)
    all_statuses = np.stack(predictions[col_eval_status].to_numpy()).flatten()  # type: ignore
    for status in all_statuses:
        if "Failed to match" in status:
            for issue in known_issues:
                if issue in status:
                    status = issue
        error_to_count[status] += 1

    error_df = pd.DataFrame(
        {col_eval_status: error_to_count.keys(), col_counts: error_to_count.values()}
    )
    error_df = error_df.sort_values(col_counts, ascending=False)

    error_pie: dict[str, int] = defaultdict(int)
    for eval_type, count in error_to_count.items():
        if "StrongPass@k used" == eval_type:
            continue
        if count < 100:
            error_pie["other"] += count
        else:
            error_pie[eval_type] = count
    error_pie_df = pd.DataFrame(
        {col_eval_status: error_pie.keys(), col_counts: error_pie.values()}
    )

    plot = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.2,
        specs=[[{"type": "table"}, {"type": "pie"}]],
    )

    plot.add_trace(
        go.Table(
            columnwidth=[500, 200],
            header=dict(
                values=error_df.columns.to_list(),
                align="left",
            ),
            cells=dict(
                values=[error_df[col_eval_status], error_df[col_counts]],
                fill_color="lavender",
                align="left",
                height=40,
            ),
        ),
        row=1,
        col=1,
    )
    plot.add_trace(
        go.Pie(
            values=error_pie_df[col_counts].tolist(),
            labels=error_pie_df[col_eval_status].tolist(),
        ),
        row=1,
        col=2,
    )
    plot.update_layout(
        height=800,
        showlegend=False,
        title_text="Evaluation Statuses",
    )
    wu.log_plot(plot_name, plot)
