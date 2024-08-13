import random

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.tokenizer import get_word_map_callable

# set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

yelp = datasets.load_dataset("yelp_polarity", split="test")
sst2 = datasets.load_dataset("sst2", split="validation")
imdb = datasets.load_dataset("imdb", split="test")

sst2 = sst2.rename_columns({"sentence": "text"})

model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

tokenizers = {}

for model_name in model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

word_map_callables = {}

for model_name in model_names:
    is_roberta = "roberta" in model_name
    word_map_callables[model_name] = get_word_map_callable(
        is_roberta=is_roberta, text_tokenizer=tokenizers[model_name]
    )


for dataset, dataset_name in [(yelp, "yelp"), (sst2, "sst2"), (imdb, "imdb")]:
    dataset = dataset.filter(lambda x: (x["label"] == 1))
    dataset = dataset.map(
        lambda x: {
            f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
            for model_name, tokenizer in tokenizers.items()
        },
        batched=True,
    )
    dataset = dataset.map(
        lambda x: {
            f"word_map_{model_name}": word_map_callables[model_name](
                x[f"input_ids_{model_name}"]
            )
            for model_name in model_names
        },
    )

    dataset = dataset.map(
        lambda x: {
            f"word_length_{model_name}": len(set(x[f"word_map_{model_name}"]))
            for model_name in model_names
        }
    )

    dataset = dataset.map(
        lambda x: {
            "word_length": max(
                [x[f"word_length_{model_name}"] for model_name in model_names]
            )
        }
    )

    dataset = dataset.map(
        lambda x: {
            "token_length": max(
                [len(x[f"input_ids_{model_name}"]) for model_name in model_names]
            )
        }
    )
    # remove examples with token length larger than 512
    dataset = dataset.filter(lambda x: x["token_length"] <= 512)
    dataset = dataset.filter(
        lambda x: all(
            x[f"word_length_{model_name}"] == x[f"word_length_{model_names[0]}"]
            for model_name in model_names
        )
    )

    dataset_small = dataset.filter(lambda x: (x["word_length"] <= 12))
    df = dataset_small.to_pandas().reset_index()

    if len(df)>0:
        df = df.rename(columns={"index": "id"})
        df[["word_length", "token_length", "label", "id", "text"]].to_csv(
            f"data/{dataset_name}_test_short.csv", index=False
    )

    df = dataset.to_pandas().reset_index()
    df = df.sample(min(1000, len(df)))  # sample 1000 random examples
    df = df.rename(columns={"index": "id"})
    df[["word_length", "token_length", "label", "id", "text"]].to_csv(
        f"data/{dataset_name}_test_long.csv", index=False
    )
