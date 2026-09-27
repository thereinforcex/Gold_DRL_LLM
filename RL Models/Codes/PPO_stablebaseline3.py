"""PPO trading agent on gold price data using Stable-Baselines3.

This script trains a Proximal Policy Optimization (PPO) agent on a custom discrete-action
trading environment built from normalized gold features (price, volume, sentiment
and model predictions). After training, the agent is evaluated on a held-out test
period, performance metrics are computed, and several diagnostic plots are generated.
"""

import numpy as np
import pandas as pd
import torch
import random
import gym
from gym import spaces
from stable_baselines3 import PPO
import plotly.graph_objects as go


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_data(df: pd.DataFrame, columns: list):
    """Normalize selected columns of a DataFrame using z-score.

    Args:
        df: Input DataFrame.
        columns: List of column names to normalize.

    Returns:
        tuple: (normalized DataFrame, means Series, stds Series)
    """
    df_norm = df.copy()
    means = df_norm[columns].mean()
    stds = df_norm[columns].std()
    df_norm[columns] = (df_norm[columns] - means) / stds
    return df_norm, means, stds


class TradingEnvCustom:
    """Custom discrete-action trading environment.

    Actions:
        0 = Buy  (long)
        1 = Sell (short)
        2 = Idle (flat)
    """

    def __init__(self, data: np.ndarray, col_means_close: float, col_stds_close: float):
        self.data = data
        self.col_means_close = col_means_close
        self.col_stds_close = col_stds_close
        self.current_step = 0
        self.n_features = data.shape[1]
        self.observation_space = None
        self.action_space = None

    def reset(self):
        """Reset the environment to the first time step."""
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action: int):
        """Execute one trading step and return the next observation, reward, done flag and info."""
        current_norm_price = self.data[self.current_step, 3]
        self.current_step += 1
        done = self.current_step >= (len(self.data) - 1)
        next_norm_price = self.data[self.current_step, 3]

        current_raw_price = (current_norm_price * self.col_stds_close) + self.col_means_close
        next_raw_price = (next_norm_price * self.col_stds_close) + self.col_means_close

        if action == 0:
            reward = (next_raw_price - current_raw_price) / current_raw_price
        elif action == 1:
            reward = (current_raw_price - next_raw_price) / current_raw_price
        elif action == 2:
            reward = 0.0
        else:
            raise ValueError("Invalid action encountered.")

        return self.data[self.current_step], reward, done, {}

    def render(self):
        """Print the current step index."""
        print("Step:", self.current_step)


class TradingGymWrapper(gym.Env):
    """Gym wrapper that exposes the custom trading environment to Stable-Baselines3."""

    def __init__(self, custom_env: TradingEnvCustom):
        super(TradingGymWrapper, self).__init__()
        self.env = custom_env
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.env.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def cumulative_return(returns) -> float:
    """Compute cumulative return from a sequence of period returns."""
    returns = np.array(returns, dtype=float)
    return np.prod(1 + returns) - 1


def max_drawdown(returns) -> float:
    """Compute maximum drawdown from a sequence of period returns."""
    returns = np.array(returns, dtype=float)
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak
    return np.min(drawdowns)


def sharpe_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Compute annualized Sharpe ratio."""
    returns = np.array(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / std_excess


def sortino_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Compute annualized Sortino ratio."""
    returns = np.array(returns, dtype=float)
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
    """Compute geometric average return per period."""
    returns = np.array(returns, dtype=float)
    n = len(returns)
    if n == 0:
        return 0.0
    return np.prod(1 + returns) ** (1 / n) - 1


def calmar_ratio(cum_return: float, max_dd: float) -> float:
    """Compute Calmar ratio (cumulative return / absolute max drawdown)."""
    if max_dd == 0:
        return np.nan
    return cum_return / abs(max_dd)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("Gold_Data_Final.csv", parse_dates=True, index_col="DateTime")
    df.sort_index(inplace=True)

    selected_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "Sentiment", "lstm_pred", "gru_pred", "transformer_pred",
    ]
    df_normalized, col_means, col_stds = normalize_data(df, selected_columns)

    basic_features = ["Open", "High", "Low", "Close", "Volume", "Sentiment", "gru_pred"]
    data_basic = df_normalized[basic_features].values.astype(np.float32)

    split_index = 727
    train_data = data_basic[:split_index]
    test_data = data_basic[split_index:]

    env_train = TradingGymWrapper(
        TradingEnvCustom(train_data, col_means["Close"], col_stds["Close"])
    )
    env_test = TradingGymWrapper(
        TradingEnvCustom(test_data, col_means["Close"], col_stds["Close"])
    )

    num_episodes = 100
    gamma = 0.80
    learning_rate = 0.00018
    steps_per_episode = train_data.shape[0] - 1
    total_timesteps = num_episodes * steps_per_episode

    print("Training PPO agent with Advanced Features")
    model_ppo = PPO(
        "MlpPolicy",
        env_train,
        gamma=gamma,
        learning_rate=learning_rate,
        verbose=1,
    )
    model_ppo.learn(total_timesteps=total_timesteps)
    model_ppo.save("ppo_advance_gold")

    print("\n--- Evaluating PPO on test data ---")
    test_actions = []
    test_rewards = []
    obs = env_test.reset()
    step_counter = 0
    done = False
    while not done:
        action, _ = model_ppo.predict(obs, deterministic=True)
        obs, reward, done, _ = env_test.step(action)
        print(f"Step: {step_counter}, Action: {action}, Reward: {reward:.4f}")
        test_actions.append(int(action))
        test_rewards.append(reward)
        step_counter += 1
    print("Evaluation complete!")

    test_df = df.iloc[split_index:].copy()
    n_test = len(test_df)
    action_map = {0: "Buy", 1: "Sell", 2: "Idle"}

    actions_str = ["Idle"] + [action_map.get(a, "Unknown") for a in test_actions]
    rewards_val = [0.0] + test_rewards

    if len(actions_str) != n_test or len(rewards_val) != n_test:
        min_len = min(len(actions_str), len(rewards_val), n_test)
        actions_str = actions_str[:min_len]
        rewards_val = rewards_val[:min_len]
        test_df = test_df.iloc[:min_len]

    test_df["Action"] = actions_str
    test_df["Reward"] = rewards_val

    print("\nEnhanced Test Table (first 5 rows):")
    print(test_df.head())

    test_df.to_csv("test_table_PPO_advance.csv")
    print("test_table_PPO_advance.csv saved.")

    periods_per_year = 252 * 6
    risk_free_rate = 0.05
    metrics_rewards = test_rewards

    cum_ret = cumulative_return(metrics_rewards)
    mdd = max_drawdown(metrics_rewards)
    sr = sharpe_ratio(metrics_rewards, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(metrics_rewards, risk_free_rate, periods_per_year)
    geom_ret = geometric_average_return(metrics_rewards)
    calmar = calmar_ratio(cum_ret, mdd)

    print("\nTest Performance Metrics (PPO - Advanced):")
    print(f"  Cumulative Return: {cum_ret:.4f}")
    print(f"  Maximum Drawdown: {mdd:.4f}")
    print(f"  Sharpe Ratio: {sr:.4f}")
    print(f"  Sortino Ratio: {sortino:.4f}")
    print(f"  Geometric Avg Return: {geom_ret:.4f}")
    print(f"  Calmar Ratio: {calmar:.4f}")

    metrics_data = {
        "Metric": [
            "Cumulative Return", "Maximum Drawdown", "Sharpe Ratio",
            "Sortino Ratio", "Geometric Avg Return", "Calmar Ratio",
        ],
        "Value": [cum_ret, mdd, sr, sortino, geom_ret, calmar],
    }
    test_metrics_df = pd.DataFrame(metrics_data)
    test_metrics_df.to_csv("test_performance_metrics_PPO_advance.csv", index=False)
    print("test_performance_metrics_PPO_advance.csv saved.")

    flat_actions = np.array(test_actions)
    buy_signals_idx = np.where(flat_actions == 0)[0]
    sell_signals_idx = np.where(flat_actions == 1)[0]
    idle_signals_idx = np.where(flat_actions == 2)[0]

    buy_signals_idx = buy_signals_idx[buy_signals_idx < len(test_df)]
    sell_signals_idx = sell_signals_idx[sell_signals_idx < len(test_df)]
    idle_signals_idx = idle_signals_idx[idle_signals_idx < len(test_df)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#3366cc", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=test_df.index[buy_signals_idx],
        y=test_df["Close"].iloc[buy_signals_idx],
        mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", color="green", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=test_df.index[sell_signals_idx],
        y=test_df["Close"].iloc[sell_signals_idx],
        mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", color="red", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=test_df.index[idle_signals_idx],
        y=test_df["Close"].iloc[idle_signals_idx],
        mode="markers",
        name="Idle",
        marker=dict(symbol="circle", color="blue", size=8),
    ))
    fig.update_layout(
        xaxis=dict(
            showline=True, showgrid=True, linecolor="black", linewidth=2,
            title_text="DateTime",
            titlefont=dict(family="Rockwell", size=12, color="black"),
        ),
        yaxis=dict(
            title_text="Price (USD)",
            titlefont=dict(family="Rockwell", size=12, color="black"),
            showline=True, showgrid=True,
            tickfont=dict(family="Rockwell", size=12, color="black"),
            linecolor="black", linewidth=2,
        ),
        template="seaborn",
        annotations=[{
            "xref": "paper", "yref": "paper", "x": 0.02, "y": 0.98,
            "xanchor": "left", "yanchor": "top",
            "text": (
                f"Test Metrics:<br>"
                f"Cumulative Return: {cum_ret:.4f}<br>"
                f"Maximum Drawdown: {mdd:.4f}<br>"
                f"Sharpe Ratio: {sr:.4f}<br>"
                f"Sortino Ratio: {sortino:.4f}<br>"
                f"Geometric Avg Return: {geom_ret:.4f}<br>"
                f"Calmar Ratio: {calmar:.4f}"
            ),
            "font": dict(family="Rockwell", size=14, color="black"),
            "bgcolor": "white", "bordercolor": "black",
            "borderwidth": 1, "borderpad": 8, "showarrow": False,
        }],
    )
    fig.show()

    initial_money = 1000
    test_df["Cumulative Money"] = initial_money * (1 + test_df["Reward"]).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df["Cumulative Money"],
        mode="lines",
        name="Cumulative Value",
        line=dict(color="#3366cc", width=3),
    ))
    fig.update_layout(
        title=dict(
            text="Cumulative Returns (Starting with $1000)",
            font=dict(family="Rockwell", size=24, color="black"),
        ),
        xaxis=dict(
            title="DateTime",
            titlefont=dict(family="Rockwell", size=12, color="black"),
            showline=True, showgrid=True, linecolor="black",
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            titlefont=dict(family="Rockwell", size=12, color="black"),
            showline=True, showgrid=True, linecolor="black",
        ),
        template="seaborn",
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df["Cumulative Money"],
        mode="lines",
        name="Cumulative Value",
        line=dict(color="#3366cc", width=3),
    ))
    buy_mask = test_df["Action"] == "Buy"
    sell_mask = test_df["Action"] == "Sell"
    idle_mask = test_df["Action"] == "Idle"
    fig.add_trace(go.Scatter(
        x=test_df.index[buy_mask],
        y=test_df["Cumulative Money"][buy_mask],
        mode="markers",
        name="Buy",
        marker=dict(symbol="triangle-up", color="green", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=test_df.index[sell_mask],
        y=test_df["Cumulative Money"][sell_mask],
        mode="markers",
        name="Sell",
        marker=dict(symbol="triangle-down", color="red", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=test_df.index[idle_mask],
        y=test_df["Cumulative Money"][idle_mask],
        mode="markers",
        name="Idle",
        marker=dict(symbol="circle", color="blue", size=8),
    ))
    fig.update_layout(
        title=dict(
            text="Cumulative Returns (Starting with $1000) and Actions",
            font=dict(family="Rockwell", size=24, color="black"),
        ),
        xaxis=dict(
            title="DateTime",
            titlefont=dict(family="Rockwell", size=12, color="black"),
            showline=True, showgrid=True, linecolor="black",
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            titlefont=dict(family="Rockwell", size=12, color="black"),
            showline=True, showgrid=True, linecolor="black",
        ),
        template="seaborn",
    )
    fig.show()



if __name__ == "__main__":
    main()