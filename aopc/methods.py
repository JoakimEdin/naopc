import math
import random
from typing import Callable, Optional
import typing
import torch
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pydantic
from rich.progress import track
import itertools
from numba import jit
from numba.typed import Dict
import numpy as np
import pandas as pd


class PerturbDataset(torch.utils.data.IterableDataset):
    def __init__(self, input_ids: torch.tensor, eos_token_id: int, bos_token_id: int, mask_token_id: int, pad_token_id: int, word_map: Optional[dict[int, list[int]]] = None):
        self.input_ids = input_ids
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.word_map = word_map


    def __iter__(self) -> tuple[int, str, torch.Tensor]:   
        has_eos = self.input_ids[0, -1].item() == self.eos_token_id
        has_bos = self.input_ids[0, 0].item() == self.bos_token_id
        num_features = len(self.word_map) if self.word_map else self.input_ids.shape[-1]
        permutation = list(range(0 + has_bos, num_features - has_eos))
        for i in range(len(permutation) + 1):
            for mask_indices in itertools.combinations(permutation, i):
                try:
                    if self.word_map:
                        mask_indices_mapped = list(itertools.chain.from_iterable(self.word_map[x] for x in mask_indices))
                    else:
                        mask_indices_mapped = list(mask_indices)
                except KeyError:
                    continue
                temp = input_ids.clone().squeeze()
                if len(mask_indices) > 0:
                    temp[mask_indices_mapped] = self.mask_token_id
                yield 0, str(sorted(mask_indices_mapped)), str(sorted(list(mask_indices))), temp
    
    def collate_fn(self, batch):
        ids, token_key, word_key, input_ids = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        return list(ids), (token_key), (word_key), input_ids

@jit()
def get_key(vector: list):
    key = ""
    for i in vector:
        key += str(i)
        key += ","
    return key


@jit()
def permutations(
    vector: list[int],
    min_order: list[int],
    max_order: list[int],
    min_max_lookup: dict[str, float],
    confident_score_dict: dict[str, float],
    step: int = 0,
):
    # if we've gotten to the end, print the permutation
    if step == len(vector):
        key = ""
        score = 0
        for i in range(0, len(vector) - 1):
            key = get_key(np.sort(vector[: i + 1]))
            score += confident_score_dict[key]

        if score > min_max_lookup["max"]:
            min_max_lookup["max"] = score
            max_order = vector

        if score < min_max_lookup["min"]:
            min_max_lookup["min"] = score
            min_order = vector

    # everything to the right of step has not been swapped yet
    for i in range(step, len(vector)):
        # copy the string (store as array)
        vector_copy = vector.copy()

        # swap the current index with the step
        vector_copy[step], vector_copy[i] = vector_copy[i], vector_copy[step]

        # recurse on the portion of the string that has not been swapped yet (now it's index will begin with step + 1)
        min_order, max_order = permutations(
            vector_copy,
            min_order,
            max_order,
            min_max_lookup,
            confident_score_dict,
            step + 1,
        )

    return min_order, max_order


def get_bounds_from_permutations(df_id: pd.DataFrame, has_eos: bool = False, has_bos: bool = False):
    # compile the functions
    d = Dict()
    min_max_lookup = Dict()
    min_max_lookup["min"] = 100.1
    min_max_lookup["max"] = 0.0
    d[""] = 0.0
    d[f"{0+has_bos},"] = 1.0
    vector = np.arange(1, 1)
    min_order = vector.copy()
    max_order = vector.copy()
    min_order, max_order = permutations(vector, min_order, max_order, min_max_lookup, d)
    d = Dict()
    min_max_lookup = Dict()


    df_id["pred"] = torch.softmax(
        torch.tensor(df_id[["positive_logit", "negative_logit"]].values), dim=1
    ).numpy()[:, 0]

    full_input_logit = df_id[df_id["word_key"] == "[]"][["id", "pred"]].rename(
        {"pred": "full_input_logit"}, axis=1
    )
    df_id = df_id.merge(full_input_logit, on="id")
    df_id["pred_diff"] = df_id["full_input_logit"] - df_id["pred"]

    for _, row in df_id.iterrows():
        key = row["word_key"][1:-1].replace(" ", "")  # Remove brackets and spaces
        if len(key) > 0:
            key += ","

        d[key] = row["pred_diff"]

    min_max_lookup["min"] = 100.1
    min_max_lookup["max"] = 0.0

    number_of_elements = len(
        df_id.iloc[df_id["word_key"].apply(len).argmax()]["word_key"].split()
    )
    vector = np.arange(0 + has_bos, number_of_elements + has_bos)
    min_order = vector.copy()
    max_order = vector.copy()

    min_order, max_order = permutations(
        vector, min_order, max_order, min_max_lookup, d
    )

    min_value = min_max_lookup["min"]
    max_value = min_max_lookup["max"]

    all_mask_diff_value = d[get_key(vector)]

    # calculate the best possible upper_limit and lower_limit
    upper_limit = (max_value + all_mask_diff_value) / number_of_elements
    lower_limit = (min_value + all_mask_diff_value) / number_of_elements

    return lower_limit, upper_limit


@dataclass
class Explanation:
    feature_importances: dict[int, int]
    remaining_features: list[int]
    previous_score: float
    cumulative_score: Optional[float]
    non_cumulative_score: float
    descending: bool
    complete: bool

class AopcSolver:

    def get_exact_bounds(self, input_ids: torch.Tensor, word_map: Optional[dict[int, list[int]]]):
        perturb_dataset = PerturbDataset(input_ids, self.eos_token_id, self.bos_token_id, self.mask_token_id, self.pad_token_id, word_map)
        perturb_dataloader = torch.utils.data.DataLoader(
            perturb_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )
        positive_logit_list = []
        negative_logit_list = []
        id_list = []
        token_key_list = []
        word_key_list = []

        has_eos, has_bos = (input_ids[0, -1].item() == self.eos_token_id), (input_ids[0, 0].item() == self.bos_token_id)
        num_features = len(word_map) if word_map else input_ids.shape[-1]

        with torch.no_grad():
            for ids, token_key, word_key, input_ids_batch in perturb_dataloader:
                logits = self.model(
                    input_ids_batch.to(self.device)
                ).logits.cpu()
                positive_logit_list.extend(logits[:, 1].tolist())
                negative_logit_list.extend(logits[:, 0].tolist())
                id_list.extend(ids)
                token_key_list.extend(token_key)
                word_key_list.extend(word_key)

        df = pd.DataFrame(
            {
                "id": id_list,
                "token_key": token_key_list,
                "word_key": word_key_list,
                "positive_logit": positive_logit_list,
                "negative_logit": negative_logit_list,
            }
        )

        lower, upper = get_bounds_from_permutations(df, has_eos=has_eos, has_bos=has_bos)
        return lower, upper


    def mask_input(self, x, value_indices, word_map=None):
        mask = torch.ones_like(x)
        try:
            if word_map is not None:
                transformed_indices = [
                    word_map[i] for i in value_indices
                ]
                value_indices = [item for sublist in transformed_indices for item in sublist]
        except KeyError:
            pass
        mask[:,value_indices] = 0
        return torch.where(
            mask == 1,
            x, 
            torch.tensor(self.mask_token_id)
        )
    
    @torch.no_grad()
    def get_prediction(self, input_ids, target_ids, device):

        temp_dataloader = torch.utils.data.DataLoader(
            input_ids,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        outs = []

        for input_ids_batch in temp_dataloader:
            input_ids_batch = input_ids_batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_pred = (
                    self.model(input_ids_batch).logits
                )  # [num_classes]
                out = torch.nn.functional.softmax(y_pred, dim=1)[:, target_ids].detach().cpu()
                outs.append(out)
        return torch.cat(outs, dim=0)


    def suggest_new_feature_importance(self, explanation: Explanation, feature_index: int):
        new_importance = len(explanation.remaining_features) - 1 if explanation.descending else len(explanation.feature_importances)
        new_feature_importances = explanation.feature_importances.copy()
        new_feature_importances[feature_index] = new_importance
        new_remaining_features = explanation.remaining_features.copy()
        new_remaining_features.remove(feature_index)
        return Explanation(
            feature_importances=new_feature_importances,
            remaining_features=new_remaining_features,
            previous_score=explanation.cumulative_score,
            cumulative_score=None,
            non_cumulative_score=0,
            descending=explanation.descending,
            complete=False,
        )


    def extend_explanation(self, explanation: Explanation):
        # For an explanation, we propose N new explanations where N is the number of remaining features
        # For each new explanation, we propose that the new feature importance is the current iteration,
        # such that for any new feature, their importance decreases or increases by 1 each iteration
        if explanation.complete:
            return [explanation]
        new_explanations = [
            self.suggest_new_feature_importance(explanation, feature_index)
            for feature_index in explanation.remaining_features
        ]
        return new_explanations
    

    def get_key_from_importances(self, feature_importance):
        return tuple(sorted(feature_importance.keys()))


    def score_explanations(self, full_input_val: float, input_ids: torch.Tensor, explanations: list[Explanation], target_ids: torch.Tensor, device: str | torch.device, word_map: dict[int, list[int]] = None, baseline: bool = False):
        complete_explanations = [explanation for explanation in explanations if explanation.complete]
        incomplete_explanations = [explanation for explanation in explanations if not explanation.complete]
        model_pass_combinations = list(set(self.get_key_from_importances(explanation.feature_importances) for explanation in incomplete_explanations))
        combination_to_score = {}
        model_inputs = torch.cat([self.mask_input(input_ids, combination, word_map) for combination in model_pass_combinations], dim=0)
        preds = self.get_prediction(model_inputs, target_ids, device)
        scores = full_input_val - preds if not baseline else preds - full_input_val
        for combination, score in zip(model_pass_combinations, scores):
            combination_to_score[combination] = score.item()
        
        new_explanations = []
        for explanation in explanations:
            key = self.get_key_from_importances(explanation.feature_importances)
            new_explanation = Explanation(
                feature_importances=explanation.feature_importances.copy(),
                remaining_features=explanation.remaining_features.copy(),
                non_cumulative_score=explanation.cumulative_score,
                cumulative_score=explanation.previous_score + combination_to_score[key],
                previous_score=explanation.previous_score,
                descending=explanat...