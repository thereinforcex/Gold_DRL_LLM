"""Comprehensive performance analysis, aggregation and visualization suite.

This module consolidates:
- Aggregation of performance-metric CSVs from multiple algorithms
- Bar-plot visualizations (individual and multi-panel)
- Cumulative portfolio-value comparison across strategies
- Sentiment-versus-action box plots
- Forecast-direction accuracy evaluation
- Prediction error metrics (MAE / RMSE / MAPE) for LSTM, GRU and Transformer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt


# ---------------------------------------------------------------------------
# 1. Aggregate performance metrics
# ---------------------------------------------------------------------------

def load_and_aggregate_metrics(
    filenames: list[str],
    desired_order: list[str],
    metrics_list: list[str],
) -> pd.DataFrame:
    """Load individual metric CSVs and build a unified summary table.

    Args:
        filenames: List of CSV paths containing metric results.
        desired_order: Desired algorithm names in the final table.
        metrics_list: Ordered list of metric names to extract.

    Returns:
        pd.DataFrame: Summary table indexed by algorithm name.
    """
    records = []
    for file in filenames:
        try:
            df = pd.read_csv(file)
            algo_mask = df["Metric"].str.strip().str.startswith("Algorithm:")
            if algo_mask.any():
                algorithm_name = (
                    df.loc[algo_mask, "Metric"].iloc[0].split("Algorithm:")[1].strip()
                )
                df = df[~algo_mask]
            else:
                algorithm_name = os.path.splitext(os.path.basename(file))[0]

            if "Value" not in df.columns:
                second_col = df.columns[1]
                df = df.rename(columns={second_col: "Value"})

            rec = {"Algorithm": algorithm_name}
            for metric in metrics_list:
                matches = df.loc[df["Metric"].str.strip() == metric, "Value"]
                rec[metric] = float(matches.iloc[0]) if not matches.empty else None
            records.append(rec)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if len(records) == len(desired_order):
        for i, rec in enumerate(records):
            rec["Algorithm"] = desired_order[i]
    else:
        print("Warning: number of records does not match desired order length.")

    summary_df = pd.DataFrame(records).set_index("Algorithm")
    return summary_df


def plot_metric_bars(summary_df: pd.DataFrame, metric: str, title_suffix: str = "") -> None:
    """Create a single Plotly bar chart for one performance metric."""
    fig = px.bar(
        summary_df.reset_index(),
        x="Algorithm",
        y=metric,
        title=f"<b>{metric} by Algorithms{title_suffix}</b>",
        color="Algorithm",
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", marker_line_width=0.5)
    fig.update_layout(
        title=dict(x=0.09, y=0.95, xanchor="left", font=dict(family="Rockwell", size=24, color="black")),
        xaxis=dict(title=dict(text="Algorithms", font=dict(family="Rockwell", size=18, color="black")), showgrid=False),
        yaxis=dict(title=dict(text=metric, font=dict(family="Rockwell", size=18, color="black")), showgrid=False),
        template="seaborn",
    )
    fig.show()


def plot_all_metrics_subplot(summary_df: pd.DataFrame, layout: str = "2x3") -> None:
    """Create a multi-panel bar chart of all six performance metrics."""
    metrics = [
        "Cumulative Return",
        "Maximum Drawdown",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Geometric Avg Return",
        "Calmar Ratio",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    if layout == "2x3":
        rows, cols = 2, 3
        figsize = (18, 12)
    else:
        rows, cols = 3, 2
        figsize = (16, 18)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(
            summary_df.index,
            summary_df[metric],
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Cumulative portfolio value comparison
# ---------------------------------------------------------------------------

def plot_cumulative_money(
    dataframes: dict[str, pd.DataFrame],
    initial_money: float = 1000.0,
) -> None:
    """Plot cumulative portfolio value for multiple strategies plus Buy & Hold.

    Args:
        dataframes: Mapping from algorithm name to its enhanced test DataFrame.
        initial_money: Starting capital.
    """
    color_dict = {
        "A2C_OHLCVSLGT": "#FF5733",
        "A2C_Diff_3&9": "#33FF57",
        "A2C_Diff_6&8": "#3357FF",
        "A2C_OHLCV": "#F1C40F",
        "A2C_OHLCVS": "#9B59B6",
        "A2C_OHLCVSG": "#1ABC9C",
        "Random_Actions": "#E67E22",
        "Buy & Hold": "#34495E",
    }

    fig = go.Figure()

    for algo_name, df in dataframes.items():
        if "Reward" in df.columns:
            reward_col = "Reward"
        elif "Random Reward" in df.columns:
            reward_col = "Random Reward"
        else:
            raise ValueError(f"No reward column found for {algo_name}")

        df = df.copy()
        df["Cumulative Money"] = initial_money * (1 + df[reward_col]).cumprod()

        fig.add_trace(
            go.Scatter(
                x=df["DateTime"],
                y=df["Cumulative Money"],
                mode="lines",
                name=algo_name,
                line=dict(width=2, color=color_dict.get(algo_name)),
            )
        )

    ref_df = dataframes.get("A2C_OHLCV", list(dataframes.values())[0]).copy()
    ref_df["Buy_Hold"] = initial_money * (ref_df["Close"] / ref_df["Close"].iloc[0])
    fig.add_trace(
        go.Scatter(
            x=ref_df["DateTime"],
            y=ref_df["Buy_Hold"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=color_dict["Buy & Hold"], width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Cumulative Money (Initial Money = 1000$)",
            font=dict(family="Rockwell", size=24, color="black"),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(text="DateTime", font=dict(family="Rockwell", size=14, color="black")),
            showgrid=False,
            showline=True,
            linecolor="black",
        ),
        yaxis=dict(
            title=dict(text="Portfolio Value ($)", font=dict(family="Rockwell", size=14, color="black")),
            showgrid=False,
            showline=True,
            linecolor="black",
        ),
        template="seaborn",
        legend=dict(title="Strategy"),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 3. Sentiment vs Action box plots
# ---------------------------------------------------------------------------

def plot_sentiment_box(file_path: str, title: str) -> None:
    """Create a colored box plot of Sentiment grouped by Action."""
    data = pd.read_csv(file_path)
    fig = px.box(
        data_frame=data,
        x="Action",
        y="Sentiment",
        category_orders={"Action": ["Buy", "Sell", "Idle"]},
        color="Action",
        color_discrete_map={"Buy": "green", "Sell": "red", "Idle": "gray"},
        title=title,
        labels={"Action": "Action", "Sentiment": "Sentiment Score"},
    )
    fig.update_layout(
        title_font=dict(family="Arial Black", size=20, color="black"),
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        font=dict(size=14),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 4. Forecast direction accuracy
# ---------------------------------------------------------------------------

def compute_direction_accuracy(
    df: pd.DataFrame,
    horizons: list[int],
    models: list[str] = ["lstm", "gru", "transformer"],
    shift_pred: bool = False,
) -> dict:
    """Compute directional accuracy of model forecasts at multiple horizons.

    Args:
        df: DataFrame containing Close and model prediction columns.
        horizons: List of forecast horizons.
        models: List of model name prefixes.
        shift_pred: If True, apply a one-step shift to the predictions.

    Returns:
        dict: Nested dictionary {model: {horizon: accuracy}}.
    """
    df_copy = df[["Close"] + [f"{m}_pred" for m in models]].copy()

    for h in horizons:
        true_col = f"True_{h}Direction"
        df_copy[true_col] = np.sign(df_copy["Close"].shift(-h) - df_copy["Close"])

        for model in models:
            pred_col = f"{model}_{h}Direction"
            if shift_pred:
                df_copy[pred_col] = np.sign(
                    df_copy[f"{model}_pred"].shift(1 - h) - df_copy[f"{model}_pred"].shift(1)
                )
            else:
                df_copy[pred_col] = np.sign(
                    df_copy[f"{model}_pred"].shift(-h) - df_copy[f"{model}_pred"]
                )

    max_h = max(horizons)
    if shift_pred:
        df_copy = df_copy.iloc[1:-max_h].copy()
    else:
        df_copy = df_copy.iloc[:-max_h].copy()

    results = {}
    for model in models:
        results[model] = {}
        for h in horizons:
            true_col = f"True_{h}Direction"
            pred_col = f"{model}_{h}Direction"
            accuracy = np.mean(df_copy[true_col] == df_copy[pred_col])
            results[model][h] = accuracy
            print(f"{model.upper()} – {h}-step direction accuracy: {accuracy * 100:.2f}%")

    return results, df_copy


# ---------------------------------------------------------------------------
# 5. Prediction error metrics
# ---------------------------------------------------------------------------

def calculate_error_metrics(y_true, y_pred) -> tuple:
    """Return MSE, RMSE, MAE, MAPE (%) and Accuracy (%)."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100
    accuracy = 100 - mape
    return mse, rmse, mae, mape, accuracy


def evaluate_prediction_errors(df: pd.DataFrame) -> None:
    """Compute and print raw and normalized error metrics for the three models."""
    df = df.copy()
    df["y_true"] = df["Close"].shift(-1)
    df = df.dropna(subset=["y_true"]).reset_index(drop=True)

    print("=== Raw Error Metrics ===")
    for name, col in [("LSTM", "lstm_pred"), ("GRU", "gru_pred"), ("Transformer", "transformer_pred")]:
        mse, rmse, mae, mape, acc = calculate_error_metrics(df["y_true"], df[col])
        print(f"{name}: MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%  Acc={acc:.2f}%")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_norm = df.copy()
    df_norm["y_true"] = scaler.fit_transform(df_norm[["y_true"]])
    for col in ["lstm_pred", "gru_pred", "transformer_pred"]:
        df_norm[col] = scaler.transform(df_norm[[col]])

    print("\n=== Normalized Error Metrics (feature_range=(-1,1)) ===")
    for name, col in [("LSTM", "lstm_pred"), ("GRU", "gru_pred"), ("Transformer", "transformer_pred")]:
        mse, rmse, mae, mape, acc = calculate_error_metrics(df_norm["y_true"], df_norm[col])
        print(f"{name}: MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%  Acc={acc:.2f}%")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    # ----- Aggregate metrics -----
    filenames = [
        "test_performance_metrics_OHLCV.csv",
        "test_performance_ohlcv_sentiment.csv",
        "test_performance_metrics_OHLCV_Sentiment_GRU.csv",
        "test_performance_metrics_OHLCV_SENTIMENT_PREDS.csv",
        "test_performance_DiffStaes_3-9.csv",
        "test_performance_DiffStaes_6-8.csv",
        "buy&hold_test_performance_metrics_OHLCV_SENTIMENT_PREDS.csv",
        "random_strategy_metrics.csv",
    ]
    desired_order = [
        "A2C_OHLCV",
        "A2C_OHLCVS",
        "A2C_OHLCVSG",
        "A2C_OHLCVSLGT",
        "A2C_Diff_3&9",
        "A2C_Diff_6&8",
        "B&H",
        "Random_Actions",
    ]
    metrics_list = [
        "Cumulative Return",
        "Maximum Drawdown",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Geometric Avg Return",
        "Calmar Ratio",
    ]

    summary_df = load_and_aggregate_metrics(filenames, desired_order, metrics_list)
    print("Summary Table of Performance Metrics:")
    print(summary_df)
    summary_df.to_csv("summary_performance_metrics.csv")
    print("summary_performance_metrics.csv saved.")

    for metric in metrics_list:
        plot_metric_bars(summary_df, metric)

    plot_all_metrics_subplot(summary_df, layout="2x3")
    plot_all_metrics_subplot(summary_df, layout="3x2")

    # ----- Cumulative money comparison -----
    dataframes = {
        "A2C_OHLCVSLGT": pd.read_csv("enhanced_test_results_OHLCV_SENTIMENT_PREDS.csv"),
        "A2C_Diff_3&9": pd.read_csv("enhanced_test_results_DiffStates_3-9.csv"),
        "A2C_Diff_6&8": pd.read_csv("enhanced_test_results_DiffStates_6-8.csv"),
        "A2C_OHLCV": pd.read_csv("enhanced_test_results_ohlcv.csv"),
        "A2C_OHLCVS": pd.read_csv("enhanced_test_results_ohlcv_sentiment.csv"),
        "A2C_OHLCVSG": pd.read_csv("enhanced_test_results_OHLCV_Sentiment_GRU.csv"),
        "Random_Actions": pd.read_csv("random_strategy_results.csv"),
    }
    for key, df in dataframes.items():
        if not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
            df["DateTime"] = pd.to_datetime(df["DateTime"])
        dataframes[key] = df.sort_values("DateTime")

    plot_cumulative_money(dataframes)

    # ----- Sentiment box plots -----
    box_files = [
        ("enhanced_test_results_OHLCV_Sentiment_GRU.csv", "Sentiment Distribution by Action in A2C_OHLCVSG"),
        ("enhanced_test_results_ohlcv_sentiment.csv", "Sentiment Distribution by Action in A2C_OHLCVS"),
        ("enhanced_test_results_DiffStates_6-8.csv", "Sentiment Distribution by Action in A2C_DiffStates6&8"),
        ("enhanced_test_results_DiffStates_3-9.csv", "Sentiment Distribution by Action in A2C_DiffStates3&9"),
        ("enhanced_test_results_OHLCV_SENTIMENT_PREDS.csv", "Sentiment Distribution by Action in A2C_OHLCVSLGT"),
    ]
    for path, title in box_files:
        if os.path.exists(path):
            plot_sentiment_box(path, title)

    # ----- Direction accuracy -----
    gold_df = pd.read_csv("Gold_Data_Final.csv", parse_dates=True, index_col="DateTime")
    gold_df.sort_index(inplace=True)

    print("\n=== Direction Accuracy (no shift) ===")
    horizons_full = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    results, df_dir = compute_direction_accuracy(gold_df, horizons_full, shift_pred=False)
    df_dir.to_csv("enhanced_gold_data_with_directions.csv")

    print("\n=== Direction Accuracy (1-step shifted predictions) ===")
    horizons_short = [3, 5, 7, 9]
    results_shift, df_dir_shift = compute_direction_accuracy(
        gold_df, horizons_short, shift_pred=True
    )
    df_dir_shift.to_csv("enhanced_gold_data_with_shifted_directions.csv")

    # ----- Prediction error metrics -----
    print("\n=== Prediction Error Metrics ===")
    evaluate_prediction_errors(gold_df)


if __name__ == "__main__":
    main()