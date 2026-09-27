"""Gold price next-step prediction using LSTM, GRU and Transformer models.

This script loads 4H gold price data, performs a chronological train/test split,
trains three sequence models, evaluates them, generates prediction plots and
saves the models together with the enriched test dataframe.
"""

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_and_prepare_data(csv_path: str = "4h_gold_price_raw.csv"):
    """Load raw gold price data and sort by datetime index.

    Args:
        csv_path: Path to the CSV file containing the 4H gold price data.

    Returns:
        pd.DataFrame: Sorted dataframe with DatetimeIndex.
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col="DateTime")
    df.sort_index(inplace=True)
    return df


def create_sequences(stock: pd.DataFrame, lookback: int, split_date: pd.Timestamp):
    """Generate sliding-window sequences and chronological train/test split.

    Args:
        stock: DataFrame containing the scaled Close prices.
        lookback: Length of the input sequence.
        split_date: Timestamp used to separate train and test periods.

    Returns:
        tuple: (x_train, y_train, x_test, y_test) as numpy arrays.
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
    """LSTM model for univariate next-step price prediction."""

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
    """GRU model for univariate next-step price prediction."""

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
    """Transformer Encoder model for univariate next-step price prediction."""

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


def main():
    df = load_and_prepare_data()

    split_date = pd.to_datetime("2024-06-10 04:00:00+00:00")
    train_df = df[:split_date]
    test_df = df[split_date:]

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    price = df[["Close"]].copy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[["Close"]].values.reshape(-1, 1))
    price["Close"] = scaler.transform(price[["Close"]].values.reshape(-1, 1))

    lookback = 9
    x_train, y_train, x_test, y_test = create_sequences(price, lookback, split_date)

    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train_lstm = torch.from_numpy(y_train).float().to(device)
    y_test_lstm = torch.from_numpy(y_test).float().to(device)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 300

    model = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        print(f"Epoch {t} MSE: {loss.item()}")
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    mse = mean_squared_error(y_test_lstm.cpu().numpy(), y_test_pred.cpu().numpy())
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_lstm.cpu().numpy(), y_test_pred.cpu().numpy())
    print("LSTM Evaluation Metrics:")
    print(pd.DataFrame({"Metric": ["MSE", "RMSE", "MAE"], "Value": [mse, rmse, mae]}))

    y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_train_inv = scaler.inverse_transform(y_train_lstm.cpu().numpy())
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())
    y_test_inv = scaler.inverse_transform(y_test_lstm.cpu().numpy())

    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred_inv) + lookback, :] = y_train_pred_inv

    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred_inv) + lookback - 1:len(price) - 1, :] = y_test_pred_inv

    original = scaler.inverse_transform(price["Close"].values.reshape(-1, 1))
    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result[0], mode="lines", name="Train prediction",
                             line=dict(color="lightgreen", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[1], mode="lines", name="Test prediction",
                             line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[2], mode="lines", name="Actual Value",
                             line=dict(color="#3366cc", width=3)))
    fig.add_shape(type="line",
                  x0=df.index.get_loc(split_date),
                  y0=result[2].min(),
                  x1=df.index.get_loc(split_date),
                  y1=result[2].max(),
                  line=dict(color="black", width=2, dash="dash"))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, showticklabels=False, linecolor="black", linewidth=2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title_text="Close (USD)", titlefont=dict(family="Rockwell", size=12, color="black"),
                   showline=True, showgrid=True, showticklabels=True, linecolor="black", linewidth=2,
                   ticks="outside", tickfont=dict(family="Rockwell", size=12, color="black")),
        showlegend=True, template="plotly",
        annotations=[dict(xref="paper", yref="paper", x=0.0, y=1.05, xanchor="left", yanchor="bottom",
                          text="Results (LSTM)", font=dict(family="Rockwell", size=26, color="black"),
                          showarrow=False)]
    )
    fig.show()

    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_gold_price_model.pth"))
    print(f"LSTM model saved to {os.path.join(model_dir, 'lstm_gold_price_model.pth')}")

    y_test_pred_df = pd.DataFrame(y_test_pred_inv, index=test_df.index[:len(y_test_pred_inv)])
    test_df = test_df.copy()
    test_df["lstm_next_price_pred"] = y_test_pred_df.shift(-1).iloc[:-1]
    test_df["lstm_next_price_pred"].iloc[-1] = test_df["Close"].iloc[-1]
    test_df.to_csv("test_df_lstm.csv")
    print("test_df_lstm.csv saved.")

    y_train_gru = y_train_lstm.clone()
    y_test_gru = y_test_lstm.clone()

    model = GRU(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 500

    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_gru)
        print(f"Epoch {t} MSE: {loss.item()}")
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    mse = mean_squared_error(y_test_gru.cpu().numpy(), y_test_pred.cpu().numpy())
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_gru.cpu().numpy(), y_test_pred.cpu().numpy())
    print("GRU Evaluation Metrics:")
    print(pd.DataFrame({"Metric": ["MSE", "RMSE", "MAE"], "Value": [mse, rmse, mae]}))

    y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())

    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred_inv) + lookback, :] = y_train_pred_inv

    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred_inv) + lookback - 1:len(price) - 1, :] = y_test_pred_inv

    original = scaler.inverse_transform(price["Close"].values.reshape(-1, 1))
    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result[0], mode="lines", name="Train prediction",
                             line=dict(color="lightgreen", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[1], mode="lines", name="Test prediction",
                             line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[2], mode="lines", name="Actual Value",
                             line=dict(color="#3366cc", width=3)))
    fig.add_shape(type="line",
                  x0=df.index.get_loc(split_date),
                  y0=result[2].min(),
                  x1=df.index.get_loc(split_date),
                  y1=result[2].max(),
                  line=dict(color="black", width=2, dash="dash"))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, showticklabels=False, linecolor="black", linewidth=2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title_text="Close (USD)", titlefont=dict(family="Rockwell", size=12, color="black"),
                   showline=True, showgrid=True, showticklabels=True, linecolor="black", linewidth=2,
                   ticks="outside", tickfont=dict(family="Rockwell", size=12, color="black")),
        showlegend=True, template="plotly",
        annotations=[dict(xref="paper", yref="paper", x=0.0, y=1.05, xanchor="left", yanchor="bottom",
                          text="Results (GRU)", font=dict(family="Rockwell", size=26, color="black"),
                          showarrow=False)]
    )
    fig.show()

    torch.save(model.state_dict(), os.path.join(model_dir, "gru_gold_price_model.pth"))
    print(f"GRU model saved to {os.path.join(model_dir, 'gru_gold_price_model.pth')}")

    y_test_pred_df = pd.DataFrame(y_test_pred_inv, index=test_df.index[:len(y_test_pred_inv)])
    test_df["gru_next_price_pred"] = y_test_pred_df.shift(-1).iloc[:-1]
    test_df["gru_next_price_pred"].iloc[-1] = test_df["Close"].iloc[-1]
    test_df.to_csv("test_df_gru.csv")
    print("test_df_gru.csv saved.")

    y_train_transformer = y_train_lstm.clone()
    y_test_transformer = y_test_lstm.clone()

    hidden_dim = 18
    num_layers = 1
    nhead = 9
    num_epochs = 1500

    model = TransformerModel(input_dim, hidden_dim, num_layers, output_dim, nhead=nhead).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        model.train()
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_transformer)
        print(f"Epoch {t} MSE: {loss.item()}")
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    training_time = time.time() - start_time
    print(f"Training time: {training_time}")

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)

    mse = mean_squared_error(y_test_transformer.cpu().numpy(), y_test_pred.cpu().numpy())
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_transformer.cpu().numpy(), y_test_pred.cpu().numpy())
    print("Transformer Evaluation Metrics:")
    print(pd.DataFrame({"Metric": ["MSE", "RMSE", "MAE"], "Value": [mse, rmse, mae]}))

    y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.cpu().numpy())

    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred_inv) + lookback, :] = y_train_pred_inv

    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred_inv) + lookback - 1:len(price) - 1, :] = y_test_pred_inv

    original = scaler.inverse_transform(price["Close"].values.reshape(-1, 1))
    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result[0], mode="lines", name="Train prediction",
                             line=dict(color="lightgreen", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[1], mode="lines", name="Test prediction",
                             line=dict(color="red", width=3)))
    fig.add_trace(go.Scatter(x=result.index, y=result[2], mode="lines", name="Actual Value",
                             line=dict(color="#3366cc", width=3)))
    fig.add_shape(type="line",
                  x0=df.index.get_loc(split_date),
                  y0=result[2].min(),
                  x1=df.index.get_loc(split_date),
                  y1=result[2].max(),
                  line=dict(color="black", width=2, dash="dash"))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True, showticklabels=False, linecolor="black", linewidth=2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title_text="Close (USD)", titlefont=dict(family="Rockwell", size=12, color="black"),
                   showline=True, showgrid=True, showticklabels=True, linecolor="black", linewidth=2,
                   ticks="outside", tickfont=dict(family="Rockwell", size=12, color="black")),
        showlegend=True, template="plotly",
        annotations=[dict(xref="paper", yref="paper", x=0.0, y=1.05, xanchor="left", yanchor="bottom",
                          text="Results (Transformer)", font=dict(family="Rockwell", size=26, color="black"),
                          showarrow=False)]
    )
    fig.show()

    torch.save(model.state_dict(), os.path.join(model_dir, "transformer_gold_price_model.pth"))
    print(f"Transformer model saved to {os.path.join(model_dir, 'transformer_gold_price_model.pth')}")

    y_test_pred_df = pd.DataFrame(y_test_pred_inv, index=test_df.index[:len(y_test_pred_inv)])
    test_df["transformer_next_price_pred"] = y_test_pred_df.shift(-1).iloc[:-1]
    test_df["transformer_next_price_pred"].iloc[-1] = test_df["Close"].iloc[-1]
    test_df.to_csv("test_df_transformer.csv")
    print("test_df_transformer.csv saved.")


if __name__ == "__main__":
    main()