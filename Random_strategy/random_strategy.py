"""Random trading strategy baseline for gold price data.

This script implements a pure random action policy (Buy / Sell / Idle) on the
test portion of the gold dataset, computes the corresponding rewards and a full
set of performance metrics, and saves both the detailed results and the summary
metrics to CSV files.
"""

import numpy as np
import pandas as pd
import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def cumulative_return(returns) -> float:
    """Compute the cumulative return over the entire period."""
    returns = np.array([float(r) for r in returns], dtype=float)
    return np.prod(1 + returns) - 1


def max_drawdown(returns) -> float:
    """Compute the maximum drawdown (largest peak-to-trough drop)."""
    returns = np.array([float(r) for r in returns], dtype=float)
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak
    return np.min(drawdowns)


def sharpe_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate the annualized Sharpe ratio."""
    returns = np.array([float(r) for r in returns], dtype=float)
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / std_excess


def sortino_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Calculate the annualized Sortino ratio using downside risk."""
    returns = np.array([float(r) for r in returns], dtype=float)
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    downside = excess_returns[excess_returns < 0]
    if downside.size < 1:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


def geometric_average_return(returns) -> float:
    """Compute the geometric average (per-period) return."""
    returns = np.array([float(r) for r in returns], dtype=float)
    n = len(returns)
    if n == 0:
        return 0.0
    return np.prod(1 + returns) ** (1 / n) - 1


def calmar_ratio(cum_return: float, max_dd: float) -> float:
    """Compute the Calmar ratio (cumulative return / absolute max drawdown)."""
    if max_dd == 0:
        return np.nan
    return cum_return / abs(max_dd)


def run_random_strategy(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
    periods_per_year: int = 252 * 6,
    risk_free_rate: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the random strategy on the test split and compute metrics.

    Args:
        df: Full gold price DataFrame indexed by DateTime.
        split_ratio: Fraction of data used for training (rest is test).
        periods_per_year: Number of periods in one year (for annualization).
        risk_free_rate: Annual risk-free rate.

    Returns:
        tuple: (test_df with Action/Reward columns, metrics_df)
    """
    split_index = int(len(df) * split_ratio)
    test_df = df.iloc[split_index:].copy()

    possible_actions = [0, 1, 2]
    random_actions = []
    random_rewards = []
    num_steps = len(test_df)

    for t in range(num_steps):
        action = np.random.choice(possible_actions)
        random_actions.append(action)

        if t < num_steps - 1:
            current_price = test_df.iloc[t]["Close"]
            next_price = test_df.iloc[t + 1]["Close"]
            price_change = (next_price - current_price) / current_price

            if action == 0:
                reward = price_change
            elif action == 1:
                reward = -price_change
            else:
                reward = 0.0
        else:
            reward = 0.0

        random_rewards.append(reward)

    test_df["Action"] = random_actions
    test_df["Reward"] = random_rewards

    random_cum_return = cumulative_return(random_rewards)
    random_max_dd = max_drawdown(random_rewards)
    random_sharpe = sharpe_ratio(random_rewards, risk_free_rate, periods_per_year)
    random_sortino = sortino_ratio(random_rewards, risk_free_rate, periods_per_year)
    random_geom_return = geometric_average_return(random_rewards)
    random_calmar = calmar_ratio(random_cum_return, random_max_dd)

    print("Random Action Strategy Metrics:")
    print(f"  Cumulative Return: {random_cum_return:.4f}")
    print(f"  Maximum Drawdown: {random_max_dd:.4f}")
    print(f"  Sharpe Ratio: {random_sharpe:.4f}")
    print(f"  Sortino Ratio: {random_sortino:.4f}")
    print(f"  Geometric Avg Return: {random_geom_return:.4f}")
    print(f"  Calmar Ratio: {random_calmar:.4f}")

    metrics_data = {
        "Metric": [
            "Cumulative Return",
            "Maximum Drawdown",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Geometric Avg Return",
            "Calmar Ratio",
        ],
        "Random Action Strategy": [
            random_cum_return,
            random_max_dd,
            random_sharpe,
            random_sortino,
            random_geom_return,
            random_calmar,
        ],
    }
    metrics_df = pd.DataFrame(metrics_data)
    return test_df, metrics_df


def main():
    set_seed(42)

    df = pd.read_csv("Gold_Data_Final.csv", parse_dates=True, index_col="DateTime")
    df.sort_index(inplace=True)

    test_df, metrics_df = run_random_strategy(df)

    metrics_df.to_csv("random_strategy_metrics.csv", index=False)
    print("random_strategy_metrics.csv saved.")

    test_df.to_csv("random_strategy_results.csv")
    print("random_strategy_results.csv saved.")


if __name__ == "__main__":
    main()