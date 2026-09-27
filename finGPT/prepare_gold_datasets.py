"""Prepare gold price datasets (4H with sentiment and daily with technical indicators).

This script:
1. Downloads 1-hour gold futures (GC=F) data and resamples it to 4-hour OHLC bars.
2. Merges the 4-hour bars with a pre-computed sentiment series.
3. Downloads daily gold data and saves it.
4. Computes common technical indicators (SMA, RSI, MACD) using stockstats.
5. Saves the indicator-enriched daily dataset and generates diagnostic plots.
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mplfinance as mpf
import stockstats


def download_and_resample_to_4h(
    ticker: str = "GC=F",
    start: str = "2024-06-10",
    end: str = "2025-01-10",
) -> pd.DataFrame:
    """Download 1-hour gold data and resample it to 4-hour OHLC bars.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date string.
        end: End date string.

    Returns:
        pd.DataFrame: Clean 4-hour OHLC dataframe (timezone-naive).
    """
    gold_1h = yf.download(ticker, start=start, end=end, interval="1h", progress=False)

    gold_1h = pd.DataFrame({
        "DateTime": gold_1h.index,
        "Open": gold_1h["Open"].values.flatten(),
        "High": gold_1h["High"].values.flatten(),
        "Low": gold_1h["Low"].values.flatten(),
        "Close": gold_1h["Close"].values.flatten(),
        "Volume": gold_1h["Volume"].values.flatten(),
    })

    gold_1h["DateTime"] = pd.to_datetime(gold_1h["DateTime"]).dt.tz_localize(None)
    gold_1h = gold_1h.set_index("DateTime")

    ohlc_dict = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    gold_4h = gold_1h[["Open", "High", "Low", "Close", "Volume"]].resample("4h").apply(ohlc_dict)
    gold_4h = gold_4h.dropna()
    return gold_4h


def merge_with_sentiment(
    gold_4h: pd.DataFrame,
    sentiment_path: str = "mean_merged_filled.csv",
) -> pd.DataFrame:
    """Merge 4-hour gold OHLC data with the sentiment series.

    Args:
        gold_4h: 4-hour OHLC dataframe.
        sentiment_path: Path to the CSV containing 'Rounded Publish Date' and 'Sentiment'.

    Returns:
        pd.DataFrame: Merged dataframe indexed by DateTime.
    """
    sentiments = pd.read_csv(sentiment_path, encoding="utf-8-sig")
    sentiments["Rounded Publish Date"] = pd.to_datetime(
        sentiments["Rounded Publish Date"]
    ).dt.tz_localize(None)

    merged = pd.merge(
        gold_4h,
        sentiments,
        left_index=True,
        right_on="Rounded Publish Date",
        how="left",
    )

    if "Unnamed: 0" in merged.columns:
        merged = merged.drop(columns=["Unnamed: 0"])

    merged = merged.set_index("Rounded Publish Date")
    merged.index.name = "DateTime"
    return merged


def download_daily_gold(
    ticker: str = "GC=F",
    start: str = "2024-06-10",
    end: str = "2025-01-10",
) -> pd.DataFrame:
    """Download daily gold futures data.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date string.
        end: End date string.

    Returns:
        pd.DataFrame: Daily OHLC dataframe indexed by Date (timezone-naive).
    """
    gold_daily = yf.download(ticker, start=start, end=end, progress=False)

    new_df = pd.DataFrame({
        "Date": gold_daily.index,
        "Open": gold_daily["Open"].values.flatten(),
        "High": gold_daily["High"].values.flatten(),
        "Low": gold_daily["Low"].values.flatten(),
        "Close": gold_daily["Close"].values.flatten(),
        "Volume": gold_daily["Volume"].values.flatten(),
    })

    new_df["Date"] = pd.to_datetime(new_df["Date"]).dt.tz_localize(None)
    new_df = new_df.set_index("Date")
    return new_df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators using stockstats.

    Args:
        df: Daily OHLC dataframe.

    Returns:
        pd.DataFrame: DataFrame enriched with RSI, SMAs and MACD columns.
    """
    stock_df = stockstats.StockDataFrame.retype(df.copy())

    stock_df["rsi_14"] = stock_df["rsi_14"]
    stock_df["close_12_sma"] = stock_df["close_12_sma"]
    stock_df["close_32_sma"] = stock_df["close_32_sma"]
    stock_df["close_200_sma"] = stock_df["close_200_sma"]
    stock_df["macd"] = stock_df["macd"]
    stock_df["macd_signal"] = stock_df["macds"]
    stock_df["macd_hist"] = stock_df["macdh"]

    return stock_df


def plot_daily_overview(df: pd.DataFrame) -> None:
    """Generate basic price and candlestick plots for the daily series.

    Args:
        df: Daily OHLC dataframe.
    """
    df["Close"].plot(figsize=(10, 7))
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.title("Gold close price data", fontsize=16)
    plt.grid(which="major", color="k", linestyle="-.", linewidth=0.5)
    plt.show()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.show()

    mpf.plot(df, type="candle", mav=(12, 32, 200), volume=True, figsize=(20, 10))


def plot_indicators(df: pd.DataFrame, stock_df: pd.DataFrame) -> None:
    """Plot Close + SMAs, RSI and MACD in a multi-panel figure.

    Args:
        df: Original daily OHLC dataframe.
        stock_df: DataFrame containing the computed indicators.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

    axes[0].plot(df.index, df["Close"], label="Close Price")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Gold Price and Indicators")
    axes[0].legend()
    axes[0].grid(which="major", color="k", linestyle="-.", linewidth=0.5)

    axes[1].plot(df.index, stock_df["close_12_sma"], label="12-Day SMA")
    axes[1].plot(df.index, stock_df["close_32_sma"], label="32-Day SMA")
    axes[1].plot(df.index, stock_df["close_200_sma"], label="200-Day SMA")
    axes[1].set_ylabel("Price")
    axes[1].set_title("SMAs")
    axes[1].legend()
    axes[1].grid(which="major", color="k", linestyle="-.", linewidth=0.5)

    axes[2].plot(df.index, stock_df["rsi_14"], label="RSI (14)")
    axes[2].axhline(70, color="red", linestyle="--")
    axes[2].axhline(30, color="green", linestyle="--")
    axes[2].set_ylabel("RSI")
    axes[2].set_title("RSI")
    axes[2].legend()
    axes[2].grid(which="major", color="k", linestyle="-.", linewidth=0.5)

    axes[3].plot(df.index, stock_df["macd"], label="MACD")
    axes[3].plot(df.index, stock_df["macd_signal"], label="MACD Signal")
    axes[3].bar(df.index, stock_df["macd_hist"], label="MACD Histogram")
    axes[3].set_ylabel("MACD")
    axes[3].set_title("MACD")
    axes[3].legend()
    axes[3].grid(which="major", color="k", linestyle="-.", linewidth=0.5)

    plt.tight_layout()
    plt.show()


def main():
    """Run the full data-preparation pipeline."""
    gold_4h = download_and_resample_to_4h()
    merged_df_sentiment = merge_with_sentiment(gold_4h, "mean_merged_filled.csv")
    merged_df_sentiment.to_csv("gold4h_price_sentiment.csv")
    print("gold4h_price_sentiment.csv saved.")

    new_df = download_daily_gold()
    new_df.to_excel("Gold_daily_row.xlsx")
    print("Gold_daily_row.xlsx saved.")

    plot_daily_overview(new_df)

    stock_df = compute_indicators(new_df)
    stock_df.to_csv("Gold_daily_indicators.csv")
    stock_df.to_excel("Gold_daily_indicators.xlsx")
    print("Gold_daily_indicators.csv and .xlsx saved.")

    plot_indicators(new_df, stock_df)


if __name__ == "__main__":
    main()