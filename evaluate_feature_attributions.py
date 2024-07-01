import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import datasets
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yelp = datasets.load_dataset(
    "csv", data_files="yelp_polarity_test_small.csv", split="train"
)

model_names = [
    "JiaqiLee/robust-bert-yelp",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]


tokenizers = {}

for model_name in model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(
        model_name, cache_dir="cache"
    )

yelp = yelp.map(
    lambda x: {
        f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir="cache"
    )
    model.to(device)
    model.eval()
    target_label = torch.tensor([1]).to(device)
    feature_attributions_df = pd.read_parquet(
        f"results/yelp_polarity_feature_attributions_{model_name.split('/')[1]}.parquet"
    )
    lower_upper_bound_comprehensiveness_and_sufficiency = pd.read_parquet(
        f"results/yelp_best_scores_{model_name.split('/')[1]}.parquet"
    )
    explanation_methods = feature_attributions_df["explanation_method"].unique()
    sufficiency_before_normalization = []
    sufficiency_after_normalization = []
    comprehensiveness_before_normalization = []
    comprehensiveness_after_normalization = []
    explanation_names = []
    id_list = []
    prob_list = []
    masked_input_list = []

    with torch.no_grad():
        for explanation_method in explanation_methods:
            for example in yelp:
                lower_upper_bound_comprehensiveness_and_sufficiency_id = (
                    lower_upper_bound_comprehensiveness_and_sufficiency[
                        lower_upper_bound_comprehensiveness_and_sufficiency["id"]
                        == example["id"]
                    ]
                )
                lower_bound_comprehensiveness = (
                    lower_upper_bound_comprehensiveness_and_sufficiency_id[
                        "sufficiency"
                    ].values[0]
                )
                upper_bound_comprehensiveness = (
                    lower_upper_bound_comprehensiveness_and_sufficiency_id[
                        "comprehensiveness"
                    ].values[0]
                )
                lower_bound_sufficiency = (
                    lower_upper_bound_comprehensiveness_and_sufficiency_id[
                        "sufficiency"
                    ].values[0]
                )
                upper_bound_sufficiency = (
                    lower_upper_bound_comprehensiveness_and_sufficiency_id[
                        "comprehensiveness"
                    ].values[0]
                )

                input_ids = (
                    torch.tensor(example[input_id_column_name]).to(device).unsqueeze(0)
                )
                full_output = (
                    model(input_ids).logits.softmax(1).squeeze(0).cpu()[1].item()
                )
                masked_input_output = (
                    lower_upper_bound_comprehensiveness_and_sufficiency_id[
                        "masked_input"
                    ].values[0]
                )

                attributions = feature_attributions_df[
                    (feature_attributions_df["id"] == example["id"])
                    & (
                        feature_attributions_df["explanation_method"]
                        == explanation_method
                    )
                ]["feature_attributions"].values[0]
                attributions = torch.from_numpy(attributions)[
                    1:-1
                ]  # ignore cls and sep token
                token_ranking = torch.argsort(attributions, descending=True) + 1
                permutation_input_ids = input_ids.clone()

                # calculate comprehensiveness
                comprehensiveness = 0
                for token in token_ranking:
                    permutation_input_ids[:, token] = mask_token_id
                    comprehensiveness += (
                        full_output
                        - model(permutation_input_ids)
                        .logits.softmax(1)
                        .squeeze(0)
                        .cpu()[1]
                        .item()
                    )

                comprehensiveness /= len(token_ranking)
                if (upper_bound_comprehensiveness - lower_bound_comprehensiveness) != 0:
                    normalized_comprehensiveness = (
                        comprehensiveness - lower_bound_comprehensiveness
                    ) / (upper_bound_comprehensiveness - lower_bound_comprehensiveness)
                else:
                    normalized_comprehensiveness = 0

                # calculate sufficiency
                sufficiency = 0
                permutation_input_ids = input_ids.clone()
                for token in token_ranking.flip(0):
                    permutation_input_ids[:, token] = mask_token_id
                    sufficiency += (
                        full_output
                        - model(permutation_input_ids)
                        .logits.softmax(1)
                        .squeeze(0)
                        .cpu()[1]
                        .item()
                    )

                sufficiency /= len(token_ranking)
                if (upper_bound_sufficiency - lower_bound_sufficiency) != 0:
                    normalized_sufficiency = (sufficiency - lower_bound_sufficiency) / (
                        upper_bound_sufficiency - lower_bound_sufficiency
                    )
                else:
                    normalized_sufficiency = 0

                explanation_names.append(explanation_method)
                comprehensiveness_before_normalization.append(comprehensiveness)
                comprehensiveness_after_normalization.append(
                    normalized_comprehensiveness
                )
                sufficiency_before_normalization.append(sufficiency)
                sufficiency_after_normalization.append(normalized_sufficiency)
                prob_list.append(full_output)
                masked_input_list.append(masked_input_output)
                id_list.append(example["id"])

    df = pd.DataFrame(
        {
            "id": id_list,
            "explanation_method": explanation_names,
            "comprehensiveness": comprehensiveness_before_normalization,
            "normalized_comprehensiveness": comprehensiveness_after_normalization,
            "sufficiency": sufficiency_before_normalization,
            "normalized_sufficiency": sufficiency_after_normalization,
            "prob": prob_list,
            "masked_input_prob": masked_input_list,
        }
    )
    df.to_csv(f"results/yelp_polarity_results_{model_name.split('/')[1]}.csv")
