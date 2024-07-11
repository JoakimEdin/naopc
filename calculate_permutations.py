import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from rich.progress import track

from dataset import PerturbDataset
from utils.tokenizer import get_word_map_callable

BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_names = ["yelp", "sst2"]
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

for dataset_name in dataset_names:
    dataset_hf = datasets.load_dataset("csv", data_files=f"data/{dataset_name}_test_short.csv", split="train")

    dataset_hf = dataset_hf.map(
        lambda x: {
            f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
            for model_name, tokenizer in tokenizers.items()
        },
        batched=True,
    )
    number_of_forward_passes = sum(dataset_hf.map(lambda x: {"2^n": math.pow(2, x["word_length"]-2)})["2^n"]) // BATCH_SIZE + 1

    for model_name in model_names:

        is_roberta = "roberta" in model_name
        word_map_callable = get_word_map_callable(is_roberta=is_roberta, text_tokenizer=tokenizers[model_name])

        dataset = PerturbDataset(dataset_hf, tokenizers[model_name].mask_token_id, tokenizers[model_name].pad_token_id, f"input_ids_{model_name}", word_map_callable=word_map_callable)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=dataset.collate_fn)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="cache")
        model.to(device)
        model.eval()

        positive_logit_list = []
        negative_logit_list = []
        id_list = []
        token_key_list = []
        word_key_list = []

        with torch.no_grad():
            for ids, token_key, word_key, input_ids_batch, attention_mask in track(dataloader, total=number_of_forward_passes, description=model_name):
                logits = model(input_ids_batch.to(device), attention_mask=attention_mask.to(device)).logits.cpu()
                positive_logit_list.extend(logits[:, 1].tolist())
                negative_logit_list.extend(logits[:, 0].tolist())
                id_list.extend(ids)
                token_key_list.extend(token_key)
                word_key_list.extend(word_key)

        df = pd.DataFrame({"id": id_list, "token_key": token_key_list, "word_key": word_key_list, "positive_logit": positive_logit_list, "negative_logit": negative_logit_list})
        df.to_parquet(f"results/permutation_outputs/{dataset_name}_{model_name.split('/')[1]}.parquet")
