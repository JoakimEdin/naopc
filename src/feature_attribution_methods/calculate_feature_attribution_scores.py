import datasets
import pandas as pd
import torch
from rich.progress import track
from transformers import AutoTokenizer

from src.feature_attribution_methods.decompx.bert import BertForSequenceClassification
from src.feature_attribution_methods.decompx.roberta import (
    RobertaForSequenceClassification,
)
from src.feature_attribution_methods.feature_attribution_methods import (
    get_attention_callable,
    get_decompx_callable,
    get_deeplift_callable,
    get_gradient_x_input_callable,
    get_integrated_gradient_callable,
    get_kernelshap_callable,
    get_lime_callable,
    get_occlusion_1_callable,
)
from src.utils.tokenizer import get_word_map_callable

BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_names = ["yelp", "sst2", "imdb"]
model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

explanation_methods = {
    "decompx": get_decompx_callable,
    "attention": get_attention_callable,
    "gradient_x_input": get_gradient_x_input_callable,
    "deeplift": get_deeplift_callable,
    "integrated_gradient": get_integrated_gradient_callable,
    "lime": get_lime_callable,
    "kernelshap": get_kernelshap_callable,
    "occlusion_1": get_occlusion_1_callable,
}

tokenizers = {}

for model_name in model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

for dataset_name in dataset_names:
    for length in ["short", "long"]:
        if (dataset_name == "imdb") and (length == "short"):
            continue  # imdb short doesn't exist

        dataset = datasets.load_dataset(
            "csv", data_files=f"data/{dataset_name}_test_{length}.csv", split="train"
        )
        dataset = dataset.map(
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
            if "roberta" in model_name:
                model = RobertaForSequenceClassification.from_pretrained(
                    model_name, cache_dir="cache"
                )
                word_map_callable = get_word_map_callable(
                    is_roberta=True, text_tokenizer=tokenizers[model_name]
                )
            else:
                model = BertForSequenceClassification.from_pretrained(
                    model_name, cache_dir="cache"
                )
                word_map_callable = get_word_map_callable(
                    is_roberta=False, text_tokenizer=tokenizers[model_name]
                )

            model.to(device)
            model.eval()
            target_label = torch.tensor([1]).to(device)

            token_attribution_list = []
            word_attribution_list = []
            id_list = []
            explanation_names = []
            word_maps = []

            for explanation_name, explanation_method in explanation_methods.items():
                explanation_method_callable = explanation_method(
                    model,
                    baseline_token_id=mask_token_id,
                    cls_token_id=start_token_id,
                    eos_token_id=end_token_id,
                )

                for example in track(
                    dataset,
                    description=f"Calculating {explanation_name}",
                    total=len(dataset),
                ):
                    input_ids = (
                        torch.tensor(example[input_id_column_name])
                        .to(device)
                        .unsqueeze(0)
                    )
                    attributions = (
                        explanation_method_callable(
                            input_ids=input_ids, target_ids=target_label, device=device
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                    word_map = word_map_callable(input_ids)
                    word_attributions = torch.zeros((word_map.max() + 1))
                    for idx, feature in enumerate(word_map):
                        word_attributions[feature] += attributions[idx]

                    token_attribution_list.append(attributions)
                    word_attribution_list.append(word_attributions.numpy())
                    id_list.append(example["id"])
                    explanation_names.append(explanation_name)
                    word_maps.append(word_map.numpy())

            df = pd.DataFrame(
                {
                    "id": id_list,
                    "token_attributions": token_attribution_list,
                    "word_attributions": word_attribution_list,
                    "explanation_method": explanation_names,
                    "word_map": word_maps,
                }
            )
            df.to_parquet(
                f"results/feature_attribution_scores/{dataset_name}_{length}_{model_name.split('/')[1]}.parquet"
            )
