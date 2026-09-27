"""Sentiment analysis of gold-related news using FinGPT (Llama-2-13B + LoRA)
and aggregation of sentiments into 4-hour intervals.

This script loads a base Llama-2-13B model with the FinGPT sentiment LoRA adapter,
classifies the sentiment of each news item, maps the results to numerical values,
and produces a 4-hour aggregated sentiment series suitable for merging with
price data.
"""

import os
import datetime
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel


def load_model_and_tokenizer(
    base_model: str = "NousResearch/Llama-2-13b-hf",
    peft_model: str = "FinGPT/fingpt-sentiment_llama2-13b_lora",
):
    """Load the base Llama-2 model and apply the FinGPT sentiment LoRA adapter.

    Args:
        base_model: Hugging Face identifier of the base model.
        peft_model: Hugging Face identifier of the Peft/LoRA adapter.

    Returns:
        tuple: (model, tokenizer) ready for inference.
    """
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
    )
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return model, tokenizer


def create_prompts(news: pd.DataFrame) -> list:
    """Create FinGPT-style sentiment classification prompts for each news row.

    Args:
        news: DataFrame containing at least 'Title' and 'Summary' columns.

    Returns:
        list: List of prompt strings.
    """
    prompts = []
    for _, row in news.iterrows():
        prompt = (
            "Instruction: What is the sentiment of this news? "
            "Please choose an answer from {negative/neutral/positive}\n"
            f"Input: Title: {row['Title']}\n"
            f"Summary: {row['Summary']}\n"
            "Answer: "
        )
        prompts.append(prompt)
    return prompts


def generate_sentiments(model, tokenizer, prompts: list) -> list:
    """Generate sentiment answers for a list of prompts.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        prompts: List of prompt strings.

    Returns:
        list: Raw decoded generation outputs.
    """
    res_sentences = []
    for prompt in prompts:
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            res = model.generate(**tokens, max_new_tokens=20, max_length=512)
        res_sentences.extend([tokenizer.decode(i, skip_special_tokens=False) for i in res])
    return res_sentences


def extract_sentiment_labels(res_sentences: list) -> list:
    """Extract and clean the sentiment label from model generations.

    Args:
        res_sentences: List of full decoded generation strings.

    Returns:
        list: Cleaned sentiment strings (positive / neutral / negative).
    """
    out_text = []
    for o in res_sentences:
        if "Answer: " in o:
            answer = o.split("Answer: ")[1]
        else:
            answer = o
        cleaned = answer.strip().lower().replace("</s>", "").strip()
        out_text.append(cleaned)
    return out_text


def map_to_numerical(out_text: list) -> list:
    """Map cleaned sentiment strings to numerical values.

    Args:
        out_text: List of cleaned sentiment labels.

    Returns:
        list: Numerical sentiments (-1, 0, 1). Unknown labels default to 0.
    """
    sentiment_mapping = {
        "positive": 1,
        "negative": -1,
        "neutral": 0,
    }
    numerical_results = []
    for result in out_text:
        numerical_results.append(sentiment_mapping.get(result, 0))
    return numerical_results


def aggregate_to_4h(sentiment_news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate news sentiments into 4-hour mean values and fill missing slots.

    Args:
        sentiment_news: DataFrame with 'Publish Date' and 'Sentiment' columns.

    Returns:
        pd.DataFrame: 4-hour indexed DataFrame with mean sentiment (forward-filled then zero-filled).
    """
    start_date = "2024-06-10 00:00:00"
    end_date = "2025-01-10 23:59:59"
    date_range = pd.date_range(start=start_date, end=end_date, freq="4H")
    df = pd.DataFrame(index=date_range)

    sentiment_news = sentiment_news.copy()
    sentiment_news["Publish Date"] = pd.to_datetime(sentiment_news["Publish Date"]).dt.tz_localize(None)
    sentiment_news["Rounded Publish Date"] = sentiment_news["Publish Date"].dt.round("4H")

    merged_df = df.merge(
        sentiment_news,
        left_index=True,
        right_on="Rounded Publish Date",
        how="left",
    )
    merged_df = merged_df.drop(columns=["Title", "Summary", "Publish Date"], errors="ignore")

    mean_merged = merged_df.groupby("Rounded Publish Date")["Sentiment"].mean()
    mean_merged_df = mean_merged.reset_index()
    mean_merged_df["Sentiment"] = mean_merged_df["Sentiment"].ffill()
    mean_merged_df["Sentiment"] = mean_merged_df["Sentiment"].fillna(0)

    return mean_merged_df


def main():
    model, tokenizer = load_model_and_tokenizer()

    news = pd.read_csv("gcf_news.csv", encoding="latin-1")
    print(news.head())

    prompts = create_prompts(news)
    res_sentences = generate_sentiments(model, tokenizer, prompts)
    out_text = extract_sentiment_labels(res_sentences)
    numerical_results = map_to_numerical(out_text)

    news["Sentiment"] = numerical_results
    print(news.head(20))

    news["Publish Date"] = pd.to_datetime(news["Publish Date"])
    news_sentiment_filename = f"news_{datetime.date.today()}.csv"
    news.to_csv(news_sentiment_filename, index=False)
    print(f"DataFrame saved to '{news_sentiment_filename}'")

    SentimentNews = pd.read_csv(news_sentiment_filename, encoding="latin-1")
    print(SentimentNews.head())

    first_date = SentimentNews["Publish Date"].min()
    last_date = SentimentNews["Publish Date"].max()
    print(f"First date and time: {first_date}")
    print(f"Last date and time: {last_date}")

    mean_merged_df = aggregate_to_4h(SentimentNews)
    mean_merged_df.to_csv("mean_merged_filled.csv", encoding="utf-8-sig", index=False)
    print("mean_merged_filled.csv saved.")

    SentimentNews["Publish Date"] = pd.to_datetime(SentimentNews["Publish Date"])
    SentimentNews["Publish Date"] = SentimentNews["Publish Date"].dt.floor("4H")
    grouped_news = SentimentNews.groupby("Publish Date").agg({"Sentiment": "mean"})
    grouped_news = grouped_news.reset_index()
    grouped_news_filename = f"grouped_news_{datetime.date.today()}.csv"
    grouped_news.to_csv(grouped_news_filename, index=False)
    print(f"DataFrame saved to '{grouped_news_filename}'")

    SentimentNews = SentimentNews.drop(columns=["Title", "Summary"], errors="ignore")
    print(SentimentNews.head(10))

    df = SentimentNews.copy()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"]).dt.date
    mean_sentiment = df.groupby("Publish Date")["Sentiment"].mean()
    mean_sentiment_df = mean_sentiment.reset_index()
    print(mean_sentiment_df)
    mean_sentiment_df.to_csv("mean_sentiment.csv", encoding="utf-8-sig", index=False)
    print("mean_sentiment.csv saved.")


if __name__ == "__main__":
    main()