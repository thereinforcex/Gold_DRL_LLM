"""Compare real versus normalized reward magnitudes for gold price data.

This script loads an enhanced test-results file, computes absolute percentage
price changes (real rewards) and the same quantity after z-score or Min-Max
normalization of the Close series, and reports summary statistics together with
the timestamps of unusually large rewards.
"""

import numpy as np
import pandas as pd


def zscore_normalize(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Apply z-score normalization to the selected columns.

    Args:
        df: Input DataFrame.
        columns: Column names to normalize.

    Returns:
        tuple: (normalized DataFrame, means, standard deviations)
    """
    df_norm = df.copy()
    means = df_norm[columns].mean()
    stds = df_norm[columns].std()
    df_norm[columns] = (df_norm[columns] - means) / stds
    return df_norm, means, stds


def minmax_normalize(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Apply Min-Max normalization to the selected columns.

    Args:
        df: Input DataFrame.
        columns: Column names to normalize.

    Returns:
        tuple: (normalized DataFrame, minima, maxima)
    """
    df_norm = df.copy()
    mins = df_norm[columns].min()
    maxs = df_norm[columns].max()
    df_norm[columns] = (df_norm[columns] - mins) / (maxs - mins)
    return df_norm, mins, maxs


def compute_rewards(df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute percentage price-change rewards on both scales.

    Args:
        df: DataFrame containing at least a 'Close' column.

    Returns:
        pd.DataFrame: Columns [Close, real_Reward, norm_Close, norm_Reward].
    """
    df_proc = df[["Close"]].copy()
    df_proc = df_proc.sort_index()

    df_z, close_mean, close_std = zscore_normalize(df_proc, ["Close"])
    df_z = df_z.rename(columns={"Close": "norm_Close"})
    df_z["Close"] = df["Close"]

    df_z["real_Reward"] = np.abs(
        (df_z["Close"].shift(-1) - df_z["Close"]) / df_z["Close"]
    )
    df_z["norm_Reward"] = np.abs(
        (df_z["norm_Close"].shift(-1) - df_z["norm_Close"]) / df_z["norm_Close"]
    )

    result = df_z[["Close", "real_Reward", "norm_Close", "norm_Reward"]].copy()
    return result, close_mean, close_std


def compute_rewards_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Same as compute_rewards but using Min-Max normalization.

    Args:
        df: DataFrame containing at least a 'Close' column.

    Returns:
        pd.DataFrame: Columns [Close, real_Reward, norm_Close, norm_Reward].
    """
    df_proc = df[["Close"]].copy()
    df_proc = df_proc.sort_index()

    df_mm, close_min, close_max = minmax_normalize(df_proc, ["Close"])
    df_mm = df_mm.rename(columns={"Close": "norm_Close"})
    df_mm["Close"] = df["Close"]

    df_mm["real_Reward"] = np.abs(
        (df_mm["Close"].shift(-1) - df_mm["Close"]) / df_mm["Close"]
    )
    df_mm["norm_Reward"] = np.abs(
        (df_mm["norm_Close"].shift(-1) - df_mm["norm_Close"]) / df_mm["norm_Close"]
    )

    df_mm["norm_Reward"] = df_mm["norm_Reward"].replace([np.inf, -np.inf], np.nan)
    df_mm["norm_Reward"] = df_mm["norm_Reward"].fillna(df_mm["norm_Reward"].mean())

    result = df_mm[["Close", "real_Reward", "norm_Close", "norm_Reward"]].copy()
    return result, close_min, close_max


def report_high_rewards(df: pd.DataFrame, threshold: float = 0.1) -> None:
    """Print indices and rows where the reward exceeds a given threshold."""
    high_real = df[df["real_Reward"] > threshold]
    high_norm = df[df["norm_Reward"] > threshold]

    print(f"\nRows with real_Reward > {threshold}: {len(high_real)}")
    if not high_real.empty:
        print(high_real)

    print(f"\nRows with norm_Reward > {threshold}: {len(high_norm)}")
    if not high_norm.empty:
        print(high_norm)


def main():
    input_file = "enhanced_test_results_OHLCV_Sentiment_GRU.csv"
    df = pd.read_csv(input_file, index_col="DateTime", parse_dates=True)

    print("=== Z-score Normalization ===")
    result_z, mean_z, std_z = compute_rewards(df)
    print("Close mean:", mean_z.values[0])
    print("Close std :", std_z.values[0])
    print("Mean real_Reward :", result_z["real_Reward"].mean())
    print("Mean norm_Reward :", result_z["norm_Reward"].mean())
    result_z.to_csv("rewards_comparison_zscore.csv")
    print("rewards_comparison_zscore.csv saved.")
    report_high_rewards(result_z)

    print("\n=== Min-Max Normalization ===")
    result_mm, min_mm, max_mm = compute_rewards_minmax(df)
    print("Close min:", min_mm.values[0])
    print("Close max:", max_mm.values[0])
    print("Mean real_Reward :", result_mm["real_Reward"].mean())
    print("Mean norm_Reward :", result_mm["norm_Reward"].mean())
    result_mm.to_csv("rewards_comparison_minmax.csv")
    print("rewards_comparison_minmax.csv saved.")
    report_high_rewards(result_mm)


if __name__ == "__main__":
    main()