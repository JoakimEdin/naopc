from typing import Optional
import typing
import torch
from dataclasses import dataclass
import itertools
from numba import jit
from numba.typed import Dict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, IterableDataset


class PerturbDataset(IterableDataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
        bos_token_id: int,
        mask_token_id: int,
        pad_token_id: int,
        word_map: dict[int, int] | dict[int, list[int]] | None = None,
    ):
        self.input_ids = input_ids
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.word_map = word_map

    def __iter__(
        self,
    ) -> typing.Generator[tuple[int, str, str, torch.Tensor], None, None]:
        has_eos = self.input_ids[0, -1].item() == self.eos_token_id
        has_bos = self.input_ids[0, 0].item() == self.bos_token_id
        num_features = len(self.word_map) if self.word_map else self.input_ids.shape[-1]
        permutation = list(range(0 + has_bos, num_features - has_eos))
        for i in range(len(permutation) + 1):
            for mask_indices in itertools.combinations(permutation, i):
                try:
                    if self.word_map:
                        mask_indices_mapped = []
                        for x in mask_indices:
                            value = self.word_map[x]
                            if isinstance(value, int):
                                mask_indices_mapped.append(value)
                            elif isinstance(value, list):
                                mask_indices_mapped.extend(value)
                            else:
                                raise TypeError(
                                    f"Unsupported type {type(value)} for word_map[{x}]"
                                )
                    else:
                        mask_indices_mapped = list(mask_indices)
                except KeyError:
                    continue
                temp = self.input_ids.clone().squeeze()
                if len(mask_indices) > 0:
                    if self.mask_token_id is not None:
                        temp[mask_indices_mapped] = self.mask_token_id
                yield (
                    0,
                    str(sorted(mask_indices_mapped)),
                    str(sorted(list(mask_indices))),
                    temp,
                )

    def collate_fn(self, batch):
        ids, token_key, word_key, input_ids = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=float(self.pad_token_id)
        )
        return list(ids), list(token_key), list(word_key), input_ids


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
            key = get_key(np.sort(vector[: i + 1]))  # type: ignore
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


def get_bounds_from_permutations(
    df_id: pd.DataFrame, has_eos: bool = False, has_bos: bool = False
):
    # compile the functions
    d = Dict()  # type: ignore
    min_max_lookup = Dict()  # type: ignore
    min_max_lookup["min"] = 100.1
    min_max_lookup["max"] = 0.0
    d[""] = 0.0
    d[f"{0+has_bos},"] = 1.0
    vector = np.arange(1, 1)
    min_order = vector.copy()
    max_order = vector.copy()
    min_order, max_order = permutations(vector, min_order, max_order, min_max_lookup, d)  # type: ignore
    d = Dict()  # type: ignore
    min_max_lookup = Dict()  # type: ignore

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

    min_order, max_order = permutations(vector, min_order, max_order, min_max_lookup, d)  # type: ignore

    min_value = min_max_lookup["min"]
    max_value = min_max_lookup["max"]

    all_mask_diff_value = d[get_key(vector)]  # type: ignore

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


def get_exact_bounds(
    model: torch.nn.Module,
    device: str | torch.device,
    input_ids: torch.Tensor,
    target_ids: int | torch.Tensor,
    eos_token_id: int,
    bos_token_id: int,
    mask_token_id: int,
    pad_token_id: int,
    word_map: dict[int, int] | dict[int, list[int]] | None,
    batch_size: int = 1024,
):
    perturb_dataset = PerturbDataset(
        input_ids, eos_token_id, bos_token_id, mask_token_id, pad_token_id, word_map
    )
    perturb_dataloader = DataLoader(
        perturb_dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    logit_list = []
    id_list = []
    token_key_list = []
    word_key_list = []

    has_eos, has_bos = (
        (input_ids[0, -1].item() == eos_token_id),
        (input_ids[0, 0].item() == bos_token_id),
    )

    with torch.no_grad():
        for ids, token_key, word_key, input_ids_batch in perturb_dataloader:
            logits = torch.softmax(
                model(input_ids_batch.to(device)).logits.cpu(), dim=1
            )
            logit_list.extend(logits[:, target_ids].squeeze().tolist())
            id_list.extend(ids)
            token_key_list.extend(token_key)
            word_key_list.extend(word_key)

    df = pd.DataFrame(
        {
            "id": id_list,
            "token_key": token_key_list,
            "word_key": word_key_list,
            "pred": logit_list,
        }
    )

    lower, upper = get_bounds_from_permutations(df, has_eos=has_eos, has_bos=has_bos)
    return lower, upper


def mask_input(x, value_indices, mask_token_id: int, word_map=None):
    mask = torch.ones_like(x)
    try:
        if word_map is not None:
            transformed_indices = [word_map[i] for i in value_indices]
            value_indices = [
                item for sublist in transformed_indices for item in sublist
            ]
    except KeyError:
        pass
    mask[:, value_indices] = 0
    return torch.where(mask == 1, x, torch.tensor(mask_token_id))


@torch.no_grad()
def get_prediction(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor | int,
    device: str | torch.device,
    batch_size: int = 1024,
):
    temp_dataloader = DataLoader(
        input_ids,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    outs = []

    for input_ids_batch in temp_dataloader:
        input_ids_batch = input_ids_batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred = model(input_ids_batch).logits  # [num_classes]
            out = (
                torch.nn.functional.softmax(y_pred, dim=1)[:, target_ids].detach().cpu()
            )
            outs.append(out)
    return torch.cat(outs, dim=0)


def suggest_new_feature_importance(explanation: Explanation, feature_index: int):
    new_importance = (
        len(explanation.remaining_features) - 1
        if explanation.descending
        else len(explanation.feature_importances)
    )
    new_feature_importances = explanation.feature_importances.copy()
    new_feature_importances[feature_index] = new_importance
    new_remaining_features = explanation.remaining_features.copy()
    new_remaining_features.remove(feature_index)
    return Explanation(
        feature_importances=new_feature_importances,
        remaining_features=new_remaining_features,
        previous_score=explanation.cumulative_score
        if explanation.cumulative_score is not None
        else 0.0,
        cumulative_score=None,
        non_cumulative_score=0,
        descending=explanation.descending,
        complete=False,
    )


def extend_explanation(explanation: Explanation):
    # For an explanation, we propose N new explanations where N is the number of remaining features
    # For each new explanation, we propose that the new feature importance is the current iteration,
    # such that for any new feature, their importance decreases or increases by 1 each iteration
    if explanation.complete:
        return [explanation]
    new_explanations = [
        suggest_new_feature_importance(explanation, feature_index)
        for feature_index in explanation.remaining_features
    ]
    return new_explanations


def get_key_from_importances(feature_importance):
    return tuple(sorted(feature_importance.keys()))


def score_explanations(
    model: torch.nn.Module,
    full_input_val: float,
    input_ids: torch.Tensor,
    explanations: list[Explanation],
    target_ids: torch.Tensor,
    mask_token_id: int,
    device: str | torch.device,
    word_map: dict[int, int] | dict[int, list[int]] | None = None,
    baseline: bool = False,
    batch_size: int = 1024,
):
    complete_explanations = [
        explanation for explanation in explanations if explanation.complete
    ]
    incomplete_explanations = [
        explanation for explanation in explanations if not explanation.complete
    ]
    model_pass_combinations = list(
        set(
            get_key_from_importances(explanation.feature_importances)
            for explanation in incomplete_explanations
        )
    )
    combination_to_score = {}
    if not model_pass_combinations:
        model_inputs = torch.empty(0, *input_ids.shape[1:], device=device)
    else:
        tensor_list = []
        for combination in model_pass_combinations:
            masked_input = mask_input(
                x=input_ids,
                value_indices=combination,
                word_map=word_map,
                mask_token_id=mask_token_id,
            )
            if not isinstance(masked_input, torch.Tensor):
                raise ValueError(
                    f"mask_input() must return a tensor, got {type(masked_input)}"
                )
            tensor_list.append(masked_input)

        model_inputs = torch.cat(tensor_list, dim=0)
    preds = get_prediction(
        model=model,
        input_ids=model_inputs,
        target_ids=target_ids,
        device=device,
        batch_size=batch_size,
    )
    scores = full_input_val - preds if not baseline else preds - full_input_val
    for combination, score in zip(model_pass_combinations, scores):
        combination_to_score[combination] = score.item()

    new_explanations = []
    for explanation in explanations:
        key = get_key_from_importances(explanation.feature_importances)
        new_explanation = Explanation(
            feature_importances=explanation.feature_importances.copy(),
            remaining_features=explanation.remaining_features.copy(),
            non_cumulative_score=explanation.cumulative_score
            if explanation.cumulative_score is not None
            else 0.0,
            cumulative_score=explanation.previous_score + combination_to_score[key],
            previous_score=explanation.previous_score,
            descending=explanation.descending,
            complete=len(explanation.remaining_features) == 0,
        )
        new_explanations.append(new_explanation)
    return new_explanations + complete_explanations


def approx_pertubation_solver_callable(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: str | torch.device,
    model: torch.nn.Module,
    mask_token_id: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    beam_size: Optional[int] = 5,
    word_map: dict[int, int] | dict[int, list[int]] | None = None,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_ids.shape[0] == 1, "Only one input at a time is supported"

    full_input_score = get_prediction(
        model=model,
        input_ids=input_ids,
        target_ids=target_ids,
        device=device,
        batch_size=batch_size,
    )

    has_cls, has_eos = (
        (input_ids[0, 0].item() == bos_token_id),
        (input_ids[0, -1].item() == eos_token_id),
    )
    if word_map is not None:
        num_features = len(word_map) - has_cls - has_eos
    else:
        num_features = input_ids.shape[1] - has_cls - has_eos

    token_range = torch.arange(0 + has_cls, num_features + has_cls)

    descending_beam = [
        Explanation(
            feature_importances={},
            remaining_features=token_range.tolist(),
            previous_score=0,
            cumulative_score=0,
            non_cumulative_score=0,
            complete=False,
            descending=True,
        )
    ]
    ascending_beam = [
        Explanation(
            feature_importances={},
            remaining_features=token_range.tolist(),
            previous_score=0,
            cumulative_score=0,
            non_cumulative_score=0,
            complete=False,
            descending=False,
        )
    ]

    max_necessary_passes = len(ascending_beam[0].remaining_features)

    for _ in range(max_necessary_passes):
        total_beam = descending_beam + ascending_beam
        explanations_to_score = []
        for explanation in total_beam:
            explanations_to_score += extend_explanation(explanation)
        new_proposed_explanations = score_explanations(
            model=model,
            full_input_val=full_input_score.item(),
            input_ids=input_ids,
            explanations=explanations_to_score,
            target_ids=target_ids,
            device=device,
            word_map=word_map,
            batch_size=batch_size,
            mask_token_id=mask_token_id,
        )
        ascending_split_index = next(
            i for i, e in enumerate(new_proposed_explanations) if not e.descending
        )
        descending_beam, ascending_beam = (
            new_proposed_explanations[:ascending_split_index],
            new_proposed_explanations[ascending_split_index:],
        )
        new_proposed_descending_explanations = sorted(
            descending_beam, key=lambda x: x.cumulative_score, reverse=True
        )
        new_proposed_ascending_explanations = sorted(
            ascending_beam, key=lambda x: x.cumulative_score, reverse=False
        )
        if beam_size is not None:
            descending_beam = new_proposed_descending_explanations[:beam_size]
            ascending_beam = new_proposed_ascending_explanations[:beam_size]
        else:
            descending_beam = new_proposed_descending_explanations
            ascending_beam = new_proposed_ascending_explanations

        total_beam = descending_beam + ascending_beam

    best_descending_explanation = descending_beam[0]
    best_ascending_explanation = ascending_beam[0]

    descending_attributions = torch.zeros(
        has_cls + len(best_descending_explanation.feature_importances) + has_eos
    )
    for (
        feature_index,
        importance,
    ) in best_descending_explanation.feature_importances.items():
        descending_attributions[feature_index] = importance

    ascending_attributions = torch.zeros(
        has_cls + len(best_ascending_explanation.feature_importances) + has_eos
    )
    for (
        feature_index,
        importance,
    ) in best_ascending_explanation.feature_importances.items():
        ascending_attributions[feature_index] = importance

    return descending_attributions, ascending_attributions


def calculate_aopc_for_attributions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_label: int,
    attributions: torch.Tensor | list[float],
    device: str | torch.device,
    mask_token_id: int,
    word_map: dict[int, int] | dict[int, list[int]] | None = None,
    descending: bool = True,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
):
    input_ids = input_ids.to(device)
    full_output = (
        model(input_ids).logits.softmax(1).squeeze(0).cpu()[target_label].item()
    )

    has_bos = input_ids[0, 0].item() == bos_token_id
    has_eos = input_ids[0, -1].item() == eos_token_id

    tokens_start = 0 + has_bos
    tokens_end = -1 if has_eos else len(input_ids[0])

    attributions = attributions[tokens_start:tokens_end]
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions)
    ranking = torch.argsort(attributions, descending=True) + has_bos
    permutation_input_ids = input_ids.clone()

    if not descending:
        ranking = ranking.flip(0)

    aopc = 0
    for entry in ranking:
        if word_map is not None:
            token_indices = word_map[int(entry.item())]
            if isinstance(token_indices, list):
                # Convert to tensor if token_indices is a list
                token_indices = torch.tensor(
                    token_indices, dtype=torch.long, device=device
                )
        else:
            token_indices = torch.tensor(
                [entry.item()], dtype=torch.long, device=device
            )

        # Index and assign mask token
        permutation_input_ids[:, token_indices] = mask_token_id

        aopc += (
            full_output
            - model(permutation_input_ids)
            .logits.softmax(1)
            .squeeze(0)
            .cpu()[target_label]
            .item()
        )

    aopc /= len(ranking)

    return aopc


def get_bounds(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_label: int,
    device: str | torch.device,
    eos_token_id: int,
    bos_token_id: int,
    mask_token_id: int,
    word_map: dict[int, int] | dict[int, list[int]] | None = None,
    normalization: typing.Literal["exact", "approx"] = "approx",
    beam_size: Optional[int] = 5,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """
    Computes the approximate or exact normalization bounds for a given input.
    """
    if normalization == "approx":
        upper_bound_order, lower_bound_order = approx_pertubation_solver_callable(
            input_ids=input_ids,
            target_ids=torch.tensor(
                [target_label]
            ),  # TODO: what if target_label is already a tensor?
            device=device,
            model=model,
            beam_size=beam_size,
            word_map=word_map,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            mask_token_id=mask_token_id,
            batch_size=batch_size,
        )
        upper_bound = calculate_aopc_for_attributions(
            model=model,
            input_ids=input_ids,
            target_label=target_label,
            attributions=upper_bound_order,
            word_map=word_map,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            mask_token_id=mask_token_id,
            device=device,
        )
        lower_bound = calculate_aopc_for_attributions(
            model=model,
            input_ids=input_ids,
            target_label=target_label,
            attributions=lower_bound_order,
            word_map=word_map,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            mask_token_id=mask_token_id,
            descending=False,
            device=device,
        )
    elif normalization == "exact":
        if input_ids.shape[-1] > 12:
            raise ValueError(
                "Exact normalization is only supported for inputs with a maximum length of 12 tokens."
            )
        lower_bound, upper_bound = get_exact_bounds(
            model=model,
            device=device,
            input_ids=input_ids,
            target_ids=target_label,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            mask_token_id=mask_token_id,
            pad_token_id=mask_token_id,
            word_map=word_map,
            batch_size=batch_size,
        )
    else:
        raise ValueError(
            f"Normalization method {normalization} is not supported. Choose 'approx' or 'exact'."
        )
    return lower_bound, upper_bound
