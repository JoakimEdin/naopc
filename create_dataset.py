import os
import math


import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from rich.progress import track

from dataset import PerturbDataset

yelp = datasets.load_dataset("yelp_polarity", split="test")

model_names = [
    "JiaqiLee/robust-bert-yelp",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

tokenizers = {}

for model_name in model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

yelp = yelp.map(
    lambda x: {
        f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
        for model_name, tokenizer in tokenizers.items()
    },
    batched=True,
)
yelp = yelp.map(
    lambda x: {
        "length": max([len(x[f"input_ids_{model_name}"]) for model_name in model_names])
    }
)

yelp = yelp.filter(lambda x: (x["length"] <= 12) and (x["label"] == 1))
yelp = yelp.sort("length", reverse=True)
yelp_df = yelp.to_pandas().reset_index()
yelp_df= yelp_df.rename(columns={"index": "id"})
yelp_df[["length", "label", "id", "text"]].to_csv("yelp_polarity_test_small.csv", index=False)

