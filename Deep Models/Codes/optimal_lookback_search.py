"""Hyperparameter search over lookback window for gold price prediction models.

Trains LSTM, GRU and Transformer models for lookback values in [5, 20],
evaluates them on a chronological test set and produces summary tables of
scaled and real-scale metrics.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_prepare_data(csv_path: str = "4h_gold_price_raw.csv"):
    """Load and sort the gold price dataframe.

    Args:
        csv_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Sorted dataframe with DatetimeIndex.
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col="DateTime")
    df.sort_index(inplace=True)
    return df


def create_sequences(stock: pd.DataFrame, lookback: int, split_date: pd.Timestamp):
    """Create sliding-window sequences and perform chronological split.

    Args:
        stock: DataFrame of scaled Close prices.
        lookback: Sequence length.
        split_date: Timestamp separating train and test.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    data_raw = stock.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)

    split_index = np.where(stock.index >= split_date)[0][0]
    train_set_size = split_index - lookback

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]
    return x_train, y_train, x_test, y_test


class LSTM(nn.Module):
    """LSTM model for next-step prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    """GRU model for next-step prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    """Transformer Encoder model for next-step prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, nhead: int = 9):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return out


def train_lstm(lookback: int, price: pd.DataFrame, scaler: MinMaxScaler, split_date: pd.Timestamp):
    """Train an LSTM model for a given lookback and return evaluation metrics.

    Args:
        lookback: Sequence length.
        price: Scaled Close price DataFrame.
        scaler: Fitted MinMaxScaler (used only for inverse transform).
        split_date: Chronological split timestamp.

    Returns:
        dict: Dictionary containing MSE, RMSE, MAE and their real-scale counterparts.
    """
    set_seed(42)
    x_train, y_train, x_test, y_test = create_sequences(price, lookback, split_date)

    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    model = LSTM(1, 32, 2, 1).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    for _ in range(500):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_test_pred.cpu().numpy()
    y_test_inv = scaler.inverse_transform(y_test_np)
    y_pred_inv = scaler.inverse_transform(y_pred_np)

    return {
        "MSE": mean_squared_error(y_test_np, y_pred_np),
        "RMSE": sqrt(mean_squared_error(y_test_np, y_pred_np)),
        "MAE": mean_absolute_error(y_test_np, y_pred_np),
        "RealMSE": mean_squared_error(y_test_inv, y_pred_inv),
        "RealRMSE": sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        "RealMAE": mean_absolute_error(y_test_inv, y_pred_inv),
    }


def train_gru(lookback: int, price: pd.DataFrame, scaler: MinMaxScaler, split_date: pd.Timestamp):
    """Train a GRU model for a given lookback and return evaluation metrics.

    Args:
        lookback: Sequence length.
        price: Scaled Close price DataFrame.
        scaler: Fitted MinMaxScaler.
        split_date: Chronological split timestamp.

    Returns:
        dict: Dictionary of scaled and real-scale metrics.
    """
    set_seed(42)
    x_train, y_train, x_test, y_test = create_sequences(price, lookback, split_date)

    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    model = GRU(1, 32, 2, 1).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(500):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_test_pred.cpu().numpy()
    y_test_inv = scaler.inverse_transform(y_test_np)
    y_pred_inv = scaler.inverse_transform(y_pred_np)

    return {
        "MSE": mean_squared_error(y_test_np, y_pred_np),
        "RMSE": sqrt(mean_squared_error(y_test_np, y_pred_np)),
        "MAE": mean_absolute_error(y_test_np, y_pred_np),
        "RealMSE": mean_squared_error(y_test_inv, y_pred_inv),
        "RealRMSE": sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        "RealMAE": mean_absolute_error(y_test_inv, y_pred_inv),
    }


def train_transformer(lookback: int, price: pd.DataFrame, scaler: MinMaxScaler, split_date: pd.Timestamp):
    """Train a Transformer model for a given lookback and return evaluation metrics.

    Args:
        lookback: Sequence length.
        price: Scaled Close price DataFrame.
        scaler: Fitted MinMaxScaler.
        split_date: Chronological split timestamp.

    Returns:
        dict: Dictionary of scaled and real-scale metrics.
    """
    set_seed(42)
    x_train, y_train, x_test, y_test = create_sequences(price, lookback, split_date)

    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    model = TransformerModel(1, 18, 1, 1, nhead=9).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(1500):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_test_pred.cpu().numpy()
    y_test_inv = scaler.inverse_transform(y_test_np)
    y_pred_inv = scaler.inverse_transform(y_pred_np)

    return {
        "MSE": mean_squared_error(y_test_np, y_pred_np),
        "RMSE": sqrt(mean_squared_error(y_test_np, y_pred_np)),
        "MAE": mean_absolute_error(y_test_np, y_pred_np),
        "RealMSE": mean_squared_error(y_test_inv, y_pred_inv),
        "RealRMSE": sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
        "RealMAE": mean_absolute_error(y_test_inv, y_pred_inv),
    }


def main():
    df = load_and_prepare_data()
    split_date = pd.to_datetime("2024-06-10 04:00:00+00:00")

    train_df = df[:split_date]
    price = df[["Close"]].copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[["Close"]].values.reshape(-1, 1))
    price["Close"] = scaler.transform(price[["Close"]].values.reshape(-1, 1))

    results = {}
    for lb in range(5, 21):
        print(f"Training for Lookback {lb}...")
        results[lb] = {}
        results[lb]["LSTM"] = train_lstm(lb, price, scaler, split_date)
        results[lb]["GRU"] = train_gru(lb, price, scaler, split_date)
        results[lb]["Transformer"] = train_transformer(lb, price, scaler, split_date)

    subcolumns = ["MSE", "RMSE", "MAE"]
    columns = pd.MultiIndex.from_product([["LSTM", "GRU", "Transformer"], subcolumns], names=["Model", "Metric"])
    table_data = []
    index_labels = []
    for lb in range(5, 21):
        row = []
        for model in ["LSTM", "GRU", "Transformer"]:
            for metric in subcolumns:
                row.append(results[lb][model][metric])
        table_data.append(row)
        index_labels.append(f"Lookback {lb}")

    df_results = pd.DataFrame(table_data, index=index_labels, columns=columns)
    df_results.index.name = "Lookback"
    print("Summary Table of Performance Metrics:")
    print(df_results)

    subcolumns_real = ["RealMSE", "RealRMSE", "RealMAE"]
    columns_real = pd.MultiIndex.from_product([["LSTM", "GRU", "Transformer"], subcolumns_real], names=["Model", "Metric"])
    R_table_data = []
    for lb in range(5, 21):
        row = []
        for model in ["LSTM", "GRU", "Transformer"]:
            for metric in subcolumns_real:
                row.append(results[lb][model][metric])
        R_table_data.append(row)

    df_results_real = pd.DataFrame(R_table_data, index=index_labels, columns=columns_real)
    df_results_real.index.name = "Lookback"
    print("Summary Table of Real Performance Metrics:")
    print(df_results_real)

    df_results.to_csv("summary_performance_metrics.csv")
    print("Summary table saved to 'summary_performance_metrics.csv'")

    df_results_real.to_csv("summary_performance_metrics_real.csv")
    print("Real-scale summary table saved to 'summary_performance_metrics_real.csv'")


if __name__ == "__main__":
    main()