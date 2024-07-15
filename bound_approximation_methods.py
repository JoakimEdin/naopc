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
    num_top_features_to_consider: Optional[int] = None,
    num_random_features_to_consider: Optional[int] = None,
    greedy_initialisation: bool = False,
    is_linear_regression: bool = False,
    word_map_callable: Optional[Callable] = None,
    **kwargs,
):

    ###############################################################
    ###############################################################
    ############# LINEAR REGRESSION SPECIFIC FUNCTIONS ############
    ###############################################################
    ###############################################################

    def get_linear_regression_prediction(input_ids, targets, device, **kwargs):
        input_ids = input_ids.float().to(device)
        outs = model(input_ids)
        return outs

    def greedy_token_truncate_linear_regression(full_input_val, input_ids, device):
        removals, removal_scores = [], []
        has_cls, has_eos = input_ids[0, 0] == cls_token_id, input_ids[0, -1] == eos_token_id


        for _ in range(0 + has_cls, input_ids.shape[1] - has_eos):
            permutations = []
            token_indices = []
            for j in range(1, input_ids.shape[1]-1):
                if input_ids[0, j] == baseline_token_id:
                    continue
                permuted_input_ids = input_ids.clone()
                permuted_input_ids[:, j] = baseline_token_id
                permutations.append(permuted_input_ids)
                token_indices.append(j)

            for example, target in zip(permutations, token_indices):
                out = get_linear_regression_prediction(example, device)
                scores.append(out)
                targets.append(target)
            scores = torch.cat(scores, dim=0)
            targets = torch.cat(targets, dim=0)
            removals.append(targets[scores.argmax().item()])
            removal_scores.append(scores.max().item())
            input_ids[:, removals[-1]] = baseline_token_id
        removals_deltas = abs(full_input_val - torch.tensor(removal_scores))
        true_index = (removals_deltas >= full_input_val.item() * 0.01).nonzero()[0].item()
        removals = removals[:true_index]
        removals_deltas = removals_deltas[:true_index]
        return removals, removals_deltas
            
    
    @torch.no_grad()
    def greedy_token_truncate(full_input_val, input_ids, device):
        removals, removal_scores = [], []
        has_cls, has_eos = input_ids[0, 0] == cls_token_id, input_ids[0, -1] == eos_token_id


        for _ in range(0 + has_cls, input_ids.shape[1] - has_eos):
            permutations = []
            token_indices = []
            for j in range(1, input_ids.shape[1]-1):
                if input_ids[0, j] == baseline_token_id:
                    continue
                permuted_input_ids = input_ids.clone()
                permuted_input_ids[:, j] = baseline_token_id
                permutations.append(permuted_input_ids)
                token_indices.append(j)


            # Create dataset of permutations and token indices
            temp_dataset = torch.utils.data.TensorDataset(
                torch.stack(permutations),
                torch.tensor(token_indices),
            )

            def collate_fn(batch):
                values, targets = [], []
                for value, target in batch:
                    values.append(value)
                    targets.append(target)
                return torch.cat(values), torch.stack(targets)
            
            temp_dataloader = torch.utils.data.DataLoader(
                temp_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            scores = []
            targets = []
            for input_ids_batch, targets_batch in temp_dataloader:
                input_ids_batch = input_ids_batch.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    y_pred = (
                        model(input_ids_batch).logits
                    )
                    out = torch.nn.functional.softmax(y_pred, dim=1)[:, 1].detach().cpu()
                    scores.append(out)
                    targets.append(targets_batch)
            scores = torch.cat(scores, dim=0)
            targets = torch.cat(targets, dim=0)
            removals.append(targets[scores.argmax().item()])
            removal_scores.append(scores.max().item())
            input_ids[:, removals[-1]] = baseline_token_id
        removals_deltas = abs(full_input_val - torch.tensor(removal_scores))
        true_index = (removals_deltas >= full_input_val.item() * 0.01).nonzero()[0].item()
        removals = removals[:true_index]
        removals_deltas = removals_deltas[:true_index]
        return removals, removals_deltas
    
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
        # Create dataloader batching the input_ids
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

    prediction_function = get_linear_regression_prediction if is_linear_regression else get_prediction
    token_truncate_function = greedy_token_truncate_linear_regression if is_linear_regression else greedy_token_truncate


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
    
    def truncate_to_feature_options(new_explanations: list[Explanation], initial_explanations: dict, n_top_features: int, n_random_features: int):

        if initial_explanations is None:
            return new_explanations
        
        initial_explanation_keys = list(initial_explanations.keys())
        initial_explanation_keys_set = set(initial_explanation_keys)

        n_top_features = n_top_features or 0
        n_random_features = n_random_features or 0

        truncated_explanations = []
        for explanation in new_explanations:
            importances_keys = list(explanation.feature_importances.keys())
            newly_added_feature = importances_keys[-1]
            if n_top_features > 0:
                top_features_search_range = n_top_features - 1
                for feature_index in explanation.feature_importances.keys():
                    if feature_index in initial_explanations:
                        top_features_search_range += 1
                top_keys = set(initial_explanation_keys[:top_features_search_range])
            else:
                top_keys = set()
            if n_random_features > 0:
                random_keys_space = initial_explanation_keys_set - top_keys - {newly_added_feature} - set(importances_keys)
                random_keys_space = list(random_keys_space)
                random.shuffle(random_keys_space)
                random_keys = random_keys_space[:n_random_features]
            else:
                random_keys = set()
            available_keys = top_keys.union(random_keys)

            if newly_added_feature in available_keys:
                truncated_explanations.append(explanation)

        
        return truncated_explanations

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
        # Convert word attributions to token attributions
        # token_attributions = torch.zeros(input_ids.shape[1])
        # if word_map:
        #     for word_index, token_indices in word_map.items():
        #         token_importance = attributions[word_index] / len(token_indices)
        #         for token_index in token_indices:
        #             token_attributions[token_index] = token_importance
        # return token_attributions

    def pertubation_solver_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
        **kwargs,
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

        if greedy_initialisation:
            removals, removal_scores = token_truncate_function(full_input_score.squeeze(0), input_ids, device)
            input_ids[:, removals] = baseline_token_id
            removed_set = set([x.item() for x in removals])
            token_range = torch.tensor([x for x in token_range if x not in removed_set])
            num_features = (input_ids.shape[1] - has_cls - has_eos) - len(removals)

        beam = [
            Explanation(
                feature_importances={},
                remaining_features=token_range.tolist(),
                previous_score=0,
                cumulative_score=0,
                non_cumulative_score=0
            )
        ]

        initial_explanations = None
 
        # One pass through all the features ensure that there are no
        # "remaining features" left in any of the explanations
        #top_n_features = [x.item()for x in token_range]
        for _ in range(num_features):
            explanations_to_score = []
            for explanation in beam:
                explanations_to_score += extend_explanation(explanation, descending)
            if num_top_features_to_consider is not None or num_random_features_to_consider is not None:
                new_proposed_explanations = truncate_to_feature_options(explanations_to_score, initial_explanations, num_top_features_to_consider, num_random_features_to_consider)
            new_proposed_explanations = score_explanations(full_input_score, input_ids, explanations_to_score, target_ids, device)
            new_proposed_explanations = sorted(new_proposed_explanations, key=lambda x: x.cumulative_score, reverse=descending)
            if initial_explanations is None:
                initial_explanations = new_proposed_explanations
                initial_explanations = {list(x.feature_importances.keys())[0]: x.cumulative_score for i, x in enumerate(initial_explanations)}

            # Pick the top N explanations and re-do the search
            beam = new_proposed_explanations[:beam_size]
        best_explanation = beam[0]
        attributions = torch.zeros(has_cls + num_features + has_eos)
        for feature_index, importance in best_explanation.feature_importances.items():
            attributions[feature_index] = importance

        # Convert word attributions to token attributions
        token_attributions = torch.zeros(input_ids.shape[1])
        if word_map:
            for word_index, token_indices in word_map.items():
                token_importance = attributions[word_index] / len(token_indices)
                for token_index in token_indices:
                    token_attributions[token_index] = token_importance
        return token_attributions


    def pertubation_solver_callable_chunked(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        """
        Experimental algorithm that partitions the input space into chunks of (currently)
        15 tokens, solves each chunk independently, and then merges the attributions based
        on the total score chunks
        """
        
        has_cls, has_eos = (input_ids[0, 0] == cls_token_id).item(), (input_ids[0, -1] == eos_token_id).item()

        end_index = -1 if has_eos else len(input_ids[1])
        cleaned_input = input_ids[:,0+has_cls:end_index]
        # Split into chunks of 15
        chunks = [cleaned_input[0, i:i + 15] for i in range(0, cleaned_input.shape[1], 15)]
        # Add cls and eos tokens to each chunk if has_cls and has_eos

        all_attributions = []

        start_index = 0
        for chunk_ids in chunks:

            # Create a tensor of mask tokens the size of the input_ids
            chunk = torch.ones_like(cleaned_input) * baseline_token_id
            # Insert the chunk ids starting at the start_index
            chunk[0, start_index:start_index + chunk_ids.shape[0]] = chunk_ids

            # Token range starts at cls_token and the start_index
            num_features = chunk_ids.shape[0]
            token_range = torch.arange(start_index + has_cls, start_index + num_features + has_cls)

            if has_cls:
                chunk = torch.cat([torch.tensor([cls_token_id]).unsqueeze(0).to(device), chunk], dim=-1)
            if has_eos:
                chunk = torch.cat([chunk, torch.tensor([eos_token_id]).unsqueeze(0).to(device)], dim=-1)

            full_input_score = prediction_function(chunk, target_ids, device)

            assert chunk.shape[0] == 1, "Only one input at a time is supported"
            beam = [
                Explanation(
                    feature_importances={},
                    remaining_features=token_range.tolist(),
                    previous_score=0,
                    cumulative_score=0,
                    non_cumulative_score=0
                )
            ]

            # One pass through all the features ensure that there are no
            # "remaining features" left in any of the explanations
            #top_n_features = [x.item()for x in token_range]
            for _ in range(num_features):
                explanations_to_score = []
                for explanation in beam:
                    explanations_to_score += extend_explanation(explanation, descending)

                new_proposed_explanations = score_explanations(full_input_score, input_ids, explanations_to_score, target_ids, device)
                new_proposed_explanations = sorted(new_proposed_explanations, key=lambda x: x.cumulative_score, reverse=descending)

                # Pick the top N explanations and re-do the search
                beam = new_proposed_explanations[:beam_size]
            best_explanation = beam[0]

            attributions = torch.zeros(has_cls + num_features + has_eos)
            for feature_index, importance in best_explanation.feature_importances.items():
                attributions[feature_index-start_index] = importance
            all_attributions.append((attributions, best_explanation.cumulative_score, start_index))
            start_index += chunk_ids.shape[0]

        # Sort all attributions by the cumulative score
        end_index = -1 if has_eos else len(input_ids[1])
        all_attributions = sorted(all_attributions, key=lambda x: x[1], reverse=descending)
        all_attributions = [(x[0][has_cls:end_index], x[2]) for x in all_attributions]
        attribution_tensor = torch.zeros((has_cls + cleaned_input.shape[1] + has_eos))
        attribution_currently = len(attribution_tensor)

        for _, (attributions, start_index) in enumerate(all_attributions):
            idx_attribution_dict = {idx: attribution for idx, attribution in enumerate(attributions)}
            idx_attribution_dict = {k: v for k, v in sorted(idx_attribution_dict.items(), key=lambda item: item[1], reverse=descending)}
            for idx, _ in idx_attribution_dict.items():
                attribution_tensor[has_cls+start_index+idx] = attribution_currently
                attribution_currently -= 1
        return attribution_tensor
    
    
    return approx_pertubation_solver_callable