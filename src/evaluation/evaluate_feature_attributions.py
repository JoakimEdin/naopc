import datasets
import pandas as pd
import torch
from rich.progress import track
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.tokenizer import get_word_idx_to_token_idxs

BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]
snli_model_names = [
    'textattack/bert-base-uncased-snli',
    'textattack/distilbert-base-cased-snli',
    'varun-v-rao/gpt2-snli-model1',
]
agnews_model_names = [
    'textattack/distilbert-base-uncased-ag-news',
    'textattack/bert-base-uncased-ag-news',
    'textattack/roberta-base-ag-news',
]

datasets_2_models = {
    "yelp": sentiment_model_names,
    "sst2": sentiment_model_names,
    "imdb": sentiment_model_names,
    "snli": snli_model_names,
    "ag_news": agnews_model_names,
}


tokenizers = {}

for model_name in sentiment_model_names + snli_model_names + agnews_model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(
        model_name, cache_dir="cache"
    )

for dataset_name in datasets_2_models.keys():
    for length in ["short", "long"]:
        if (dataset_name in {"imdb", "snli", "ag_news"}) and (length == "short"):
            continue
        dataset = datasets.load_dataset(
            "csv", data_files=f"data/{dataset_name}_test_{length}.csv", split="train"
        )
        if dataset_name == "snli":
            dataset = dataset.map(lambda x: {"text": x["premise"] + " " + x["hypothesis"]})
        dataset = dataset.map(
            lambda x: {
                f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
                for model_name, tokenizer in tokenizers.items()
            },
            batched=True,
        )

        for model_name in datasets_2_models[dataset_name]:

            mask_token_id = tokenizers[model_name].mask_token_id if not "gpt2" in model_name else tokenizers[model_name].eos_token_id
            pad_token_id = tokenizers[model_name].pad_token_id if not "gpt2" in model_name else tokenizers[model_name].eos_token_id
            start_token_id = tokenizers[model_name].cls_token_id
            end_token_id = tokenizers[model_name].sep_token_id

            input_id_column_name = f"input_ids_{model_name}"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, cache_dir="cache"
            )
            model.to(device)
            model.eval()
            feature_attributions_df = pd.read_parquet(
                f"results/feature_attribution_scores/{dataset_name}_{length}_{model_name.split('/')[1]}.parquet"
            )
            explanation_methods = feature_attributions_df["explanation_method"].unique()
            sufficiency_list = []
            comprehensiveness_list = []
            explanation_names = []
            id_list = []
            prob_list = []

            with torch.no_grad():
                for explanation_method in explanation_methods:
                    for example in track(
                        dataset,
                        description=f"Evaluating {explanation_method}",
                        total=len(dataset),
                    ):                        
                        target_label = example["label"]
                        input_ids = (
                            torch.tensor(example[input_id_column_name])
                            .to(device)
                            .unsqueeze(0)
                        )
                        target_label = torch.tensor(example["label"]).to(device)
                        full_output = (
                            model(input_ids)
                            .logits.softmax(1)
                            .squeeze(0)
                            .cpu()[target_label]
                            .item()
                        )

                        attributions_for_eample = feature_attributions_df[
                            (feature_attributions_df["id"] == example["id"])
                            & (
                                feature_attributions_df["explanation_method"]
                                == explanation_method
                            )
                        ]

                        token_attributions = attributions_for_eample[
                            "token_attributions"
                        ].values[0]
                        word_attributions = attributions_for_eample[
                            "word_attributions"
                        ].values[0]
                        word_map = attributions_for_eample["word_map"].values[0]

                        if 1 not in word_map:
                            word_map = word_map - 1
                            word_map[0] = 0

                        has_bos = input_ids[0, 0] == start_token_id
                        has_eos = input_ids[0, -1] == end_token_id

                        tokens_start = 0 + has_bos
                        tokens_end = -1 if has_eos else len(input_ids[0])

                        word_map_dict = get_word_idx_to_token_idxs(word_map)

                        word_attributions = torch.from_numpy(word_attributions)[
                            tokens_start:tokens_end
                        ]  # ignore cls and sep token
                        word_ranking = (
                            torch.argsort(word_attributions, descending=True) + has_bos
                        )
                        permutation_input_ids = input_ids.clone()

                        comprehensiveness = 0
                        for word in word_ranking:
                            token_indices = word_map_dict[word.item()]

                            permutation_input_ids[:, token_indices] = mask_token_id
                            comprehensiveness += (
                                full_output
                                - model(permutation_input_ids)
                                .logits.softmax(1)
                                .squeeze(0)
                                .cpu()[target_label]
                                .item()
                            )

                        comprehensiveness /= len(word_ranking)

                        # calculate sufficiency
                        sufficiency = 0
                        permutation_input_ids = input_ids.clone()
                        for word in word_ranking.flip(0):
                            token_indices = word_map_dict[word.item()]
                            permutation_input_ids[:, token_indices] = mask_token_id
                            sufficiency += (
                                full_output
                                - model(permutation_input_ids)
                                .logits.softmax(1)
                                .squeeze(0)
                                .cpu()[target_label]
                                .item()
                            )

                        sufficiency /= len(word_ranking)

                        explanation_names.append(explanation_method)
                        comprehensiveness_list.append(comprehensiveness)
                        sufficiency_list.append(sufficiency)
                        prob_list.append(full_output)
                        id_list.append(example["id"])

            df = pd.DataFrame(
                {
                    "id": id_list,
                    "explanation_method": explanation_names,
                    "comprehensiveness": comprehensiveness_list,
                    "sufficiency": sufficiency_list,
                    "prob": prob_list,
                }
            )
            df.to_csv(
                f"results/aopc_scores_{length}/{dataset_name}_{model_name.split('/')[1]}.csv"
            )
