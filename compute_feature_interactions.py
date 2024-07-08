import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import datasets
import pandas as pd
import torch
from transformers import AutoTokenizer
from rich.progress import track

from transformers import AutoTokenizer

from decompx.bert import BertForSequenceClassification
from decompx.roberta import RobertaForSequenceClassification
from archipelago.explainer import Archipelago


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
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, truncation=True)

yelp = yelp.map(
    lambda x: {
        f"input_ids_{model_name}": tokenizer(x["text"], max_length=512)["input_ids"]
        for model_name, tokenizer in tokenizers.items()
    },
    batched=True,
)

for model_name in model_names:
    mask_token_id = tokenizers[model_name].mask_token_id
    pad_token_id = tokenizers[model_name].pad_token_id
    start_token_id = tokenizers[model_name].cls_token_id
    end_token_id = tokenizers[model_name].sep_token_id

    input_id_column_name = f"input_ids_{model_name}"
    if "roberta" in model_name:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, cache_dir="cache"
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name, cache_dir="cache"
        )

    model.to(device)
    model.eval()
    target_label = 1

    id_list = []
    score_dicts = []

    for example in track(yelp, description=f"Calculating feature interactions", total=len(yelp)):
        input_ids = (
            torch.tensor(example[input_id_column_name])
        )
        baseline_ids = input_ids.clone()
        baseline_ids[1:-1] = mask_token_id

        apgo = Archipelago(model, input=input_ids, baseline=baseline_ids, output_indices=target_label, verbose=False, cls_token_id=start_token_id, eos_token_id=end_token_id)
        # start_time = time.time()
        interactions_dict = apgo.archdetect()

        apgo_invert = Archipelago(model, input=baseline_ids, baseline=input_ids, output_indices=target_label, verbose=False, cls_token_id=start_token_id, eos_token_id=end_token_id)
        inverted_interactions_dict = apgo_invert.archdetect()
        # print(f"Time taken: {time.time() - start_time}")

        pair_scores_dict = {tuple(pair): {} for pair in interactions_dict["pairwise_effects"].keys()}

        for pair, score in interactions_dict["pairwise_effects"].items():
            pair_scores_dict[tuple(pair)]["effect_score"] = score

        for pair, score in interactions_dict["interactions"]:
            pair_scores_dict[tuple(pair)]["interaction_score"] = score

        for pair, score in inverted_interactions_dict["pairwise_effects"].items():
            pair_scores_dict[tuple(pair)]["inverted_effect_score"] = score

        for pair, score in inverted_interactions_dict["interactions"]:
            pair_scores_dict[tuple(pair)]["inverted_interaction_score"] = score


        id_list.append(example["id"])
        score_dicts.append(pair_scores_dict)

    scores = []

    df = pd.DataFrame(
        {
            "id": id_list,
            "feature_interactions": score_dicts,
        }
    )

    df.to_parquet(
        f"results/yelp_polarity_feature_interactions_{model_name.split('/')[1]}.parquet"
    )
