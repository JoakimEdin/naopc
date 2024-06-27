import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from rich.progress import track

from dataset import PerturbDataset

BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yelp = datasets.load_dataset("csv", data_files="yelp_polarity_test_small.csv", split="train")

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
number_of_forward_passes = sum(yelp.map(lambda x: {"2^n": math.pow(2, x["length"]-2)})["2^n"]) // BATCH_SIZE + 1

for model_name in model_names:

    dataset = PerturbDataset(yelp, tokenizers[model_name].mask_token_id, tokenizers[model_name].pad_token_id, f"input_ids_{model_name}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=dataset.collate_fn)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="cache")
    model.to(device)
    model.eval()

    positive_logit_list = []
    negative_logit_list = []
    id_list = []
    key_list = []

    
    with torch.no_grad():
        for ids, key, input_ids_batch, attention_mask in track(dataloader, total=number_of_forward_passes, description=model_name):
            logits = model(input_ids_batch.to(device), attention_mask=attention_mask.to(device)).logits.cpu()
            positive_logit_list.extend(logits[:, 1].tolist())
            negative_logit_list.extend(logits[:, 0].tolist())
            id_list.extend(ids)
            key_list.extend(key)

    df = pd.DataFrame({"id": id_list, "key": key_list, "positive_logit": positive_logit_list, "negative_logit": negative_logit_list})
    df.to_parquet(f"results/yelp_polarity_permutations_{model_name.split('/')[1]}.parquet")
