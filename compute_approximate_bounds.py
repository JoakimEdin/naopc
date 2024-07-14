import math
import os
import warnings
from typing import Optional

import datasets
import pandas as pd
import torch
from transformers import AutoTokenizer
from rich.progress import track
import argparse
import time

from transformers import AutoTokenizer

from utils.tokenizer import get_word_idx_to_token_idxs, get_word_map_callable
from decompx.bert import BertForSequenceClassification
from decompx.roberta import RobertaForSequenceClassification
from bound_approximation_methods import (
    get_comprehensiveness_solver_callable,
    get_sufficiency_solver_callable,
)

@torch.no_grad()
def main(
        dataset_name,
        model_name,
        beam_size,
        dataset_length: str = "short",
        explanation_attributions: Optional[str] = False,
        use_exact_limits: bool = False,
        
):
    print("Running script with the following parameters:")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Beam size: {beam_size}")
    print(f"Dataset length: {dataset_length}")
    print(f"Explanation attributions: {explanation_attributions}")
    print(f"Use exact limits: {use_exact_limits}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_exact_limits and dataset_length == "long":
        warnings.warn("Exact limits are not available for long datasets.")
        return


    get_sufficiency_solver =  get_sufficiency_solver_callable
    get_comprehensiveness_solver = get_comprehensiveness_solver_callable

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if (dataset_name == "imdb") and (dataset_length == "short"):
        warnings.warn("IMDB short dataset does not exist. Using long dataset instead.")
        return 

    dataset = datasets.load_dataset("csv", data_files=f"data/{dataset_name}_test_{dataset_length}.csv", split="train")
    dataset = dataset.map(
        lambda x: {
            f"input_ids": tokenizer(x["text"])["input_ids"]
        },
        batched=True,
    )

    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    start_token_id = tokenizer.cls_token_id
    end_token_id = tokenizer.sep_token_id

    if "roberta" in model_name:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, cache_dir="cache"
        )
        word_map_callable = get_word_map_callable(
            is_roberta=True, text_tokenizer=tokenizer
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name, cache_dir="cache"
        )
        word_map_callable = get_word_map_callable(
            is_roberta=False, text_tokenizer=tokenizer
        )

    model.to(device)
    model.eval()
    target_label = torch.tensor([1]).to(device)

    if explanation_attributions is not None:
        preprocessing_step = "feature_attributions"
        attributions_df = pd.read_parquet(
            f"results/feature_attribution_scores/{dataset_name}_{dataset_length}_{model_name.split('/')[1]}.parquet"
        )
    else:
        preprocessing_step = None

    if use_exact_limits:
        exact_limits_df = pd.read_parquet(
            f"results/aopc_limits_exact/{dataset_name}_{model_name.split('/')[1]}.parquet"
        )

    id_list = []
    word_maps = []
    upper_word_attribution_list = []
    lower_word_attribution_list = []
    upper_delta_list = []
    lower_delta_list = []
    preprocessing_list = []
    explanation_method = []
    processing_time = []

    sufficiency_solver_callable = get_sufficiency_solver(
        model,
        baseline_token_id=mask_token_id,
        cls_token_id=start_token_id,
        eos_token_id=end_token_id,
        word_map_callable=word_map_callable,
        preprocessing_step=preprocessing_step,
        beam_size=beam_size
    )
    comprehensiveness_solver_callable = get_comprehensiveness_solver(
        model,
        baseline_token_id=mask_token_id,
        cls_token_id=start_token_id,
        eos_token_id=end_token_id,
        word_map_callable=word_map_callable,
        preprocessing_step=preprocessing_step,
        beam_size=beam_size
    )

    i = 0
    for example in track(
        dataset, 
        description=f"Approximating bounds...", 
        total=len(dataset)
    ):
        print(f"Processing example {i}")
        i += 1
        input_ids = (
            torch.tensor(example["input_ids"]).to(device).unsqueeze(0)
        )
        if explanation_attributions is not None:
            attributions_for_example = attributions_df[
                attributions_df["id"] == example["id"]
            ]
            attributions = attributions_for_example[attributions_for_example["explanation_method"] == explanation_attributions]["word_attributions"].values[0]
        else:
            attributions = None

        start = time.time()
        sufficiency_attributions_out = (
            sufficiency_solver_callable(
                input_ids=input_ids, target_ids=target_label, device=device, attributions=attributions,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        comprehensiveness_attributions_out = (
            comprehensiveness_solver_callable(
                input_ids=input_ids, target_ids=target_label, device=device, attributions=attributions,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        total_time = time.time() - start

        word_map = word_map_callable(input_ids)

        full_output = (
            model(input_ids).logits.softmax(1).squeeze(0).cpu()[1].item()
        )
        
        word_map_dict = get_word_idx_to_token_idxs(word_map)

        permutation_input_ids = input_ids.clone()
        comp_aopc = 0
        comprehensiveness_attributions = torch.from_numpy(comprehensiveness_attributions_out[1:-1])
        sorted_comprehensiveness_attributions = torch.argsort(comprehensiveness_attributions, descending=True) + 1

        permutation_input_ids = input_ids.clone()
        batch_tensors = []
        for word in sorted_comprehensiveness_attributions:
            token_indices = word_map_dict[word.item()]
            permutation_input_ids[:, token_indices] = mask_token_id
            batch_tensors.append(permutation_input_ids.clone())

        batch_tensors = torch.cat(batch_tensors, dim=0)
        preds = model(batch_tensors).logits.softmax(1)[:, 1]
        full_output_minus_preds = full_output - preds

        comp_aopc = full_output_minus_preds.sum().item() / len(sorted_comprehensiveness_attributions)

        permutation_input_ids = input_ids.clone()
        suff_aopc = 0
        sufficiency_attributions = torch.from_numpy(sufficiency_attributions_out[1:-1])
        sorted_sufficiency_attributions = torch.argsort(sufficiency_attributions, descending=False) + 1

        permutation_input_ids = input_ids.clone()
        batch_tensors = []
        for word in sorted_sufficiency_attributions:
            token_indices = word_map_dict[word.item()]
            permutation_input_ids[:, token_indices] = mask_token_id
            batch_tensors.append(permutation_input_ids.clone())

        batch_tensors = torch.cat(batch_tensors, dim=0)
        preds = model(batch_tensors).logits.softmax(1)[:, 1]
        full_output_minus_preds = full_output - preds

        suff_aopc = full_output_minus_preds.sum().item() / len(sorted_sufficiency_attributions)

        
        if use_exact_limits:
            limits = exact_limits_df[exact_limits_df["id"] == example["id"]]
            suff_lower = limits["lower_limit"].values[0]
            comp_upper = limits["upper_limit"].values[0]

            upper_delta = abs(comp_upper - comp_aopc)
            lower_delta = abs(suff_lower - suff_aopc)
        else:
            upper_delta = math.nan
            lower_delta = math.nan

        id_list.append(example["id"])
        word_maps.append(word_map.numpy())
        upper_word_attribution_list.append(comprehensiveness_attributions.numpy())
        lower_word_attribution_list.append(sufficiency_attributions.numpy())
        upper_delta_list.append(upper_delta)
        lower_delta_list.append(lower_delta)
        preprocessing_list.append(preprocessing_step)
        explanation_method.append(explanation_attributions)
        processing_time.append(total_time)

    df = pd.DataFrame(
        {
            "id": id_list,
            "word_map": word_maps,
            "comprehensiveness_upper_word_attributions": upper_word_attribution_list,
            "sufficiency_lower_word_attributions": lower_word_attribution_list,
            "comprehensiveness_upper_delta": upper_delta_list,
            "sufficiency_lower_delta": lower_delta_list,
            "preprocessing": preprocessing_list,
            "explanation_method": explanation_method,
            "processing_time": processing_time,
        }
    )
    #clamp_string = "clamp" if clamp else "no_clamp"
    preprocesing_string = "no_preprocessing" if preprocessing_step is None else preprocessing_step + "_" + explanation_attributions
    df.to_parquet(
        f"results/aopc_limits_approx/{dataset_name}_{dataset_length}_{preprocesing_string}_beam_size_{str(beam_size)}_{model_name.split('/')[1]}.parquet"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--model_name", type=str, default="textattack/bert-base-uncased-SST-2")
    parser.add_argument("--explanation_attributions", type=str, required=False)
    parser.add_argument("--use_exact_limits", type=bool, default=True)
    parser.add_argument("--dataset_length", type=str, default="short")
    parser.add_argument("--cuda_id", type=str, default="7")
    parser.add_argument("--beam_size", type=int, default=50)
    args = parser.parse_args()
    if args.dataset_length == "long":
        warnings.warn("Long datasets are not supported for exact bound comparisons.")
        args.use_exact_limits = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        explanation_attributions=args.explanation_attributions,
        use_exact_limits=args.use_exact_limits,
        dataset_length=args.dataset_length,
        beam_size=args.beam_size
    )
