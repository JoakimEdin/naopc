import random
from typing import Callable, Optional
import torch

from utils.tokenizer import get_word_idx_to_token_idxs

def get_aopc_solver_callable(
        *args,
        **kwargs,
    ):
    return get_pertubation_solver_callable(
        *args,
        **kwargs,
    )

@torch.no_grad()
def get_pertubation_solver_callable(
    model: torch.nn.Module | Callable,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    beam_size: int = 50,
    batch_size: int = 1024,
    preprocessing_step: Optional[str] = None,
    is_linear_regression: bool = False,
    word_map_callable: Optional[Callable] = None,
    **kwargs,
):
    
    class Explanation:
        def __init__(self, feature_importances, remaining_features, previous_score, cumulative_score, non_cumulative_score, descending, complete):
            self.feature_importances = feature_importances
            self.remaining_features = remaining_features
            self.previous_score = previous_score
            self.cumulative_score = cumulative_score
            self.non_cumulative_score = non_cumulative_score
            self.descending = descending
            self.complete = complete


    def mask_input(x, value_indices, word_map=None):
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
            torch.tensor(baseline_token_id)
        )

    def unmask_input(x, value_indices, word_map=None):
        mask = torch.zeros_like(x)
        if word_map is not None:
            transformed_indices = [
                word_map[i] for i in value_indices
            ]
            value_indices = [item for sublist in transformed_indices for item in sublist]
        mask[:,value_indices] = 1
        mask[:,[0,-1]] = 1
        return torch.where(
            mask == 1,
            x,
            torch.tensor(baseline_token_id),
        )
    
    @torch.no_grad()
    def get_prediction(input_ids, target_ids, device):

        temp_dataloader = torch.utils.data.DataLoader(
            input_ids,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        outs = []

        for input_ids_batch in temp_dataloader:
            input_ids_batch = input_ids_batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_pred = (
                    model(input_ids_batch).logits
                )  # [num_classes]
                out = torch.nn.functional.softmax(y_pred, dim=1)[:, target_ids].detach().cpu()
                outs.append(out)
        return torch.cat(outs, dim=0)

    prediction_function = get_prediction


    def suggest_new_feature_importance(explanation: Explanation, feature_index: int):
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


    def score_explanations(full_input_val: float, input_ids: torch.Tensor, explanations: list[Explanation], target_ids: torch.Tensor, device: str | torch.device, mask_fn: Callable = mask_input, word_map: dict[int, list[int]] = None, baseline: bool = False):
        complete_explanations = [explanation for explanation in explanations if explanation.complete]
        incomplete_explanations = [explanation for explanation in explanations if not explanation.complete]
        model_pass_combinations = list(set(get_key_from_importances(explanation.feature_importances) for explanation in incomplete_explanations))
        combination_to_score = {}
        model_inputs = torch.cat([mask_fn(input_ids, combination, word_map) for combination in model_pass_combinations], dim=0)
        preds = prediction_function(model_inputs, target_ids, device)
        scores = full_input_val - preds if not baseline else preds - full_input_val
        for combination, score in zip(model_pass_combinations, scores):
            combination_to_score[combination] = score.item()
        
        new_explanations = []
        for explanation in explanations:
            key = get_key_from_importances(explanation.feature_importances)
            new_explanation = Explanation(
                feature_importances=explanation.feature_importances.copy(),
                remaining_features=explanation.remaining_features.copy(),
                non_cumulative_score=explanation.cumulative_score,
                cumulative_score=explanation.previous_score + combination_to_score[key],
                previous_score=explanation.previous_score,
                descending=explanation.descending,
                complete=len(explanation.remaining_features) == 0
            )
            new_explanations.append(new_explanation)
        return new_explanations + complete_explanations

    def approx_pertubation_solver_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attributions: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        
        assert input_ids.shape[0] == 1, "Only one input at a time is supported"

        full_input_score = prediction_function(input_ids, target_ids, device)

            
        has_cls, has_eos = (input_ids[0, 0] == cls_token_id).item(), (input_ids[0, -1] == eos_token_id).item()
        if word_map_callable is not None:
            word_map = get_word_idx_to_token_idxs(word_map_callable(input_ids))
            num_features = len(word_map) - has_cls - has_eos
        else:
            num_features = input_ids.shape[1] - has_cls - has_eos

        token_range = torch.arange(0 + has_cls, num_features + has_cls)

        if preprocessing_step == "feature_attributions":
            assert attributions is not None, "Attributions must be provided for feature attribution"
            ATTRIBUTIONS_THRESHOLD = 0.05
            weak_features = []
            for i, attr in enumerate(attributions[1:-1], start=1):
                #if abs(attr) < ATTRIBUTIONS_THRESHOLD:
                if attr < ATTRIBUTIONS_THRESHOLD:
                    weak_features.append((i, attr))

            # Descending/ascending is inverted now since we want the lowest impact to the respective scores
            weak_feature_importances_descending = sorted(weak_features, key=lambda x: x[1], reverse=False)
            weak_feature_importances_descending = {feature: -i for i, (feature, _) in enumerate(weak_feature_importances_descending, start=1)}
            weak_feature_importances_ascending = sorted(weak_features, key=lambda x: x[1], reverse=True)
            weak_feature_importances_ascending = {feature: -i for i, (feature, _) in enumerate(weak_feature_importances_ascending, start=1)}
            weak_features = set([x[0] for x in weak_features])

            descending_explanation = Explanation(
                        feature_importances={},
                        remaining_features=list(set(token_range.tolist()) - weak_features),
                        previous_score=0,
                        non_cumulative_score=0,
                        cumulative_score=0,
                        complete=False,
                        descending=True,
                    )
            descending_beam = score_explanations(full_input_score, input_ids, [descending_explanation], target_ids, device, word_map=word_map)

            ascending_explanation = Explanation(
                        feature_importances=weak_feature_importances_ascending,
                        remaining_features=list(set(token_range.tolist()) - weak_features),
                        previous_score=0,
                        non_cumulative_score=0,
                        cumulative_score=0,
                        complete=False,
                        descending=False,
                    )
            ascending_beam = score_explanations(full_input_score, input_ids, [ascending_explanation], target_ids, device, word_map=word_map)
                
        
        elif preprocessing_step is None:
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
            new_proposed_explanations = score_explanations(full_input_score, input_ids, explanations_to_score, target_ids, device, word_map=word_map)
            ascending_split_index = next(i for i, e in enumerate(new_proposed_explanations) if e.descending == False)
            descending_beam, ascending_beam = new_proposed_explanations[:ascending_split_index], new_proposed_explanations[ascending_split_index:]
            new_proposed_descending_explanations = sorted(descending_beam, key=lambda x: x.cumulative_score, reverse=True)
            new_proposed_ascending_explanations = sorted(ascending_beam, key=lambda x: x.cumulative_score, reverse=False)
            descending_beam = new_proposed_descending_explanations[:beam_size]
            ascending_beam = new_proposed_ascending_explanations[:beam_size]

            total_beam = descending_beam + ascending_beam

        best_descending_explanation = descending_beam[0]
        best_ascending_explanation = ascending_beam[0]


        # If we are calculating comprehensiveness we add the weak features to the explanation
        # at the end. If we are calculating sufficiency, they were already prepended
        if preprocessing_step:
            best_descending_explanation.feature_importances.update(weak_feature_importances_descending)
            
        descending_attributions = torch.zeros(has_cls + len(best_descending_explanation.feature_importances) + has_eos)
        for feature_index, importance in best_descending_explanation.feature_importances.items():
            descending_attributions[feature_index] = importance

        ascending_attributions = torch.zeros(has_cls + len(best_ascending_explanation.feature_importances) + has_eos)
        for feature_index, importance in best_ascending_explanation.feature_importances.items():
            ascending_attributions[feature_index] = importance

        return descending_attributions, ascending_attributions
    
    return approx_pertubation_solver_callable