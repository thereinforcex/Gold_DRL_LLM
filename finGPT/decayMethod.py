"""Create a 4-hour gold price dataset enriched with exponentially decayed news sentiment.

This script:
1. Loads news sentiment data.
2. Aligns sentiments to a regular 4-hour grid.
3. Fills missing sentiment values using exponential decay from the last valid observation.
4. Downloads gold futures (GC=F) 1-hour data via yfinance and resamples it to 4-hour OHLC.
5. Merges the two series and saves the final dataset.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def load_sentiment_news(path: str = "news_sentiment.csv") -> pd.DataFrame:
    """Load the news sentiment CSV and print its date range.

    Args:
        path: Path to the CSV file containing at least 'Publish Date' and 'Sentiment'.

    Returns:
        pd.DataFrame: Loaded sentiment news dataframe.
    """
    sentiment_news = pd.read_csv(path, encoding="latin-1")
    first_date = sentiment_news["Publish Date"].min()
    last_date = sentiment_news["Publish Date"].max()
    print(f"First date and time: {first_date}")
    print(f"Last date and time: {last_date}")
    return sentiment_news


def create_4h_grid(start: str = "2024-06-10 00:00:00", end: str = "2025-01-10 23:59:59") -> pd.DataFrame:
    """Create an empty DataFrame indexed by a regular 4-hour date range.

    Args:
        start: Start datetime string.
        end: End datetime string.

    Returns:
        pd.DataFrame: Empty dataframe with 4-hour DatetimeIndex.
    """
    date_range = pd.date_range(start=start, end=end, freq="4H")
    return pd.DataFrame(index=date_range)


def align_sentiment_to_4h(sentiment_news: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Align individual news sentiments onto the 4-hour grid by rounding timestamps.

    Args:
        sentiment_news: DataFrame with 'Publish Date' and 'Sentiment' columns.
        grid: Empty 4-hour indexed DataFrame.

    Returns:
        pd.DataFrame: Merged dataframe containing Rounded Publish Date and Sentiment.
    """
    sentiment_news = sentiment_news.copy()
    sentiment_news["Publish Date"] = pd.to_datetime(sentiment_news["Publish Date"]).dt.tz_localize(None)
    sentiment_news["Rounded Publish Date"] = sentiment_news["Publish Date"].dt.round("4H")

    merged = grid.merge(
        sentiment_news,
        left_index=True,
        right_on="Rounded Publish Date",
        how="left",
    )
    merged = merged.drop(columns=["Title", "Summary", "Publish Date"], errors="ignore")
    return merged


def compute_mean_sentiment(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean sentiment per 4-hour slot.

    Args:
        merged: DataFrame produced by align_sentiment_to_4h.

    Returns:
        pd.DataFrame: DataFrame with columns ['Rounded Publish Date', 'Sentiment'].
    """
    mean_merged = merged.groupby("Rounded Publish Date")["Sentiment"].mean()
    return mean_merged.reset_index()


def apply_exponential_decay(df: pd.DataFrame, decay_rate: float = 0.03) -> pd.DataFrame:
    """Fill NaN sentiment values using exponential decay from the last valid observation.

    The formula used is: S(t) = S(0) * exp(-decay_rate * t)
    where t is the number of 4-hour steps since the last valid value.

    Args:
        df: DataFrame containing a 'Sentiment' column that may contain NaNs.
        decay_rate: Positive decay coefficient (lambda).

    Returns:
        pd.DataFrame: DataFrame with decayed sentiment values. Remaining leading NaNs are set to 0.
    """
    df = df.copy()
    last_valid_index = None
    last_valid_value = None

    for i in range(len(df)):
        if pd.notna(df.loc[i, "Sentiment"]):
            last_valid_index = i
            last_valid_value = df.loc[i, "Sentiment"]
        else:
            if last_valid_value is not None:
                t = i - last_valid_index
                df.loc[i, "Sentiment"] = last_valid_value * np.exp(-decay_rate * t)

    df["Sentiment"] = df["Sentiment"].fillna(0)
    return df


def download_and_resample_gold(
    ticker: str = "GC=F",
    start: str = "2024-06-10",
    end: str = "2025-01-10",
) -> pd.DataFrame:
    """Download 1-hour gold futures data and resample it to 4-hour OHLC bars.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date string.
        end: End date string.

    Returns:
        pd.DataFrame: 4-hour OHLC dataframe indexed by DateTime (timezone-naive).
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
    return gold_4h


def merge_price_and_sentiment(gold_4h: pd.DataFrame, sentiments: pd.DataFrame) -> pd.DataFrame:
    """Merge 4-hour gold OHLC data with the decayed sentiment series.

    Args:
        gold_4h: 4-hour OHLC dataframe.
        sentiments: DataFrame with 'Rounded Publish Date' and 'Sentiment'.

    Returns:
        pd.DataFrame: Merged dataframe indexed by DateTime, with rows containing NaN Close removed.
    """
    sentiments = sentiments.copy()
    sentiments["Rounded Publish Date"] = pd.to_datetime(sentiments["Rounded Publish Date"])

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
    merged = merged.dropna(subset=["Close"])
    return merged


def main(decay_rate: float = 0.03):
    """Run the full pipeline: sentiment alignment, exponential decay, gold download and final merge.

    Args:
        decay_rate: Exponential decay coefficient used to fill missing sentiment values.
    """
    sentiment_news = load_sentiment_news("news_sentiment.csv")
    grid = create_4h_grid()
    merged = align_sentiment_to_4h(sentiment_news, grid)
    mean_merged_df = compute_mean_sentiment(merged)
    mean_merged_df = apply_exponential_decay(mean_merged_df, decay_rate=decay_rate)

    mean_merged_df.to_csv("mean_merged_decay.csv", encoding="utf-8-sig", index=False)
    print("mean_merged_decay.csv saved.")

    gold_4h = download_and_resample_gold()
    final_df = merge_price_and_sentiment(gold_4h, mean_merged_df)

    final_df.to_csv("gold4h_price_sentiment_decay.csv")
    print("gold4h_price_sentiment_decay.csv saved.")
    print(final_df.head())


if __name__ == "__main__":
    main(decay_rate=0.03)