from typing import Callable, Optional

import captum
import torch

from decompx.decompx_utils import DecompXConfig
import random

Explainer = Callable[[torch.Tensor, torch.Tensor, str | torch.device], torch.Tensor]

CONFIGS = DecompXConfig(
    include_biases=True,
    bias_decomp_type="absdot",
    include_LN1=True,
    include_FFN=True,
    FFN_approx_type="GeLU_ZO",
    include_LN2=True,
    aggregation="vector",
    include_classifier_w_pooler=True,
    tanh_approx_type="ZO",
    output_all_layers=True,
    output_attention=None,
    output_res1=None,
    output_LN1=None,
    output_FFN=None,
    output_res2=None,
    output_encoder=None,
    output_aggregated="norm",
    output_pooler="norm",
    output_classifier=True,
)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None
    ):
        return self.model(input_ids).logits


def create_baseline_input(
    input_ids: torch.Tensor,
    baseline_token_id: int = 50_000,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """Create baseline input for a given input

    Args:
        input_ids (torch.Tensor): Input ids to create baseline input for
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        torch.Tensor: Baseline input
    """
    baseline = torch.ones_like(input_ids) * baseline_token_id
    baseline[:, 0] = cls_token_id
    baseline[:, -1] = eos_token_id
    return baseline


def embedding_attributions_to_token_attributions(
    attributions: torch.Tensor,
) -> torch.Tensor:
    """Convert embedding attributions to token attributions.

    Args:
        attributions (torch.Tensor): Embedding Attributions,

    Returns:
        torch.Tensor: Token attributions
    """

    return torch.norm(attributions, p=2, dim=-1)

def get_comprehensiveness_solver_callable(
        *args,
        **kwargs,
    ):
    return get_pertubation_solver_callable(
        *args,
        **kwargs,
        descending=True
    )

def get_sufficiency_solver_callable(
        *args,
        **kwargs,
    ):
    return get_pertubation_solver_callable(
        *args,
        **kwargs,
        descending=False
    )

@torch.no_grad()
def get_pertubation_solver_callable(
    model: torch.nn.Module | Callable,
    descending: bool,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    beam_size: int = 50,
    batch_size: int = 1024,
    num_top_features_to_consider: Optional[int] = None,
    num_random_features_to_consider: Optional[int] = None,
    greedy_initialisation: bool = False,
    is_linear_regression: bool = False,
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
        def __init__(self, feature_importances, remaining_features, previous_score, cumulative_score, non_cumulative_score):
            self.feature_importances = feature_importances
            self.remaining_features = remaining_features
            self.previous_score = previous_score
            self.cumulative_score = cumulative_score
            self.non_cumulative_score = non_cumulative_score


    def mask_input(x, value_indices):
        mask = torch.ones_like(x)
        mask[:,value_indices] = 0
        return torch.where(
            mask == 1,
            x, 
            torch.tensor(baseline_token_id)
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


    def suggest_new_feature_importance(explanation: Explanation, feature_index: int, new_importance: int):
        new_feature_importances = explanation.feature_importances.copy()
        new_feature_importances[feature_index] = new_importance
        new_remaining_features = explanation.remaining_features.copy()
        new_remaining_features.remove(feature_index)
        return Explanation(
            feature_importances=new_feature_importances,
            remaining_features=new_remaining_features,
            previous_score=explanation.cumulative_score,
            cumulative_score=None,
            non_cumulative_score=0
        )


    def extend_explanation(explanation: Explanation, descending: bool):
        # For an explanation, we propose N new explanations where N is the number of remaining features
        # For each new explanation, we propose that the new feature importance is the current iteration,
        # such that for any new feature, their importance decreases or increases by 1 each iteration
        current_importance = len(explanation.remaining_features) - 1 if descending else len(explanation.feature_importances)
        new_explanations = [
            suggest_new_feature_importance(explanation, feature_index, current_importance)
            for feature_index in explanation.remaining_features
        ]
        return new_explanations
    

    def get_key_from_importances(feature_importance):
        return tuple(sorted(feature_importance.keys()))


    def score_explanations(full_input_val: float, input_ids: torch.Tensor, explanations: list[Explanation], target_ids: torch.Tensor, device: str | torch.device):
        model_pass_combinations = list(set(get_key_from_importances(explanation.feature_importances) for explanation in explanations))
        combination_to_score = {}
        model_inputs = torch.cat([mask_input(input_ids, combination) for combination in model_pass_combinations], dim=0)
        preds = prediction_function(model_inputs, target_ids, device)
        scores = full_input_val - preds
        for combination, score in zip(model_pass_combinations, scores):
            combination_to_score[combination] = score.item()
        
        new_explanations = []
        for explanation in explanations:
            key = get_key_from_importances(explanation.feature_importances)
            explanation.cumulative_score = explanation.previous_score + combination_to_score[key]
            explanation.non_cumulative_score = combination_to_score[key]
            new_explanations.append(explanation)
        return new_explanations
    
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

    def pertubation_solver_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        
        assert input_ids.shape[0] == 1, "Only one input at a time is supported"

        full_input_score = prediction_function(input_ids, target_ids, device)
            
        has_cls, has_eos = (input_ids[0, 0] == cls_token_id).item(), (input_ids[0, -1] == eos_token_id).item()
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
        return attributions


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
    
    
    return pertubation_solver_callable

def get_decompx_callable(
    model: torch.nn.Module,
    **kwargs,
) -> Explainer:
    model.eval()
    
    @torch.no_grad()
    def decompx_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            (
                logits,
                decompx_last_layer_outputs,
                decompx_all_layers_outputs,
            ) = model(
                input_ids,
                output_attentions=False,
                return_dict=False,
                output_hidden_states=False,
                decompx_config=CONFIGS,
            )

        attributions = decompx_last_layer_outputs.classifier.detach().squeeze(0)
        return attributions[:, target_ids]

    return decompx_callable


@torch.no_grad()
def get_occlusion_1_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    **kwargs,
) -> Explainer:
    def occlusion_1_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        attributions = torch.zeros(sequence_length, num_classes)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred = (
                model(input_ids).logits[:, target_ids].detach().cpu().squeeze(0)
            )  # [num_classes]

        for idx in range(sequence_length):
            permuted_input_ids = input_ids.clone()
            permuted_input_ids[:, idx] = baseline_token_id
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_permute = (
                    model(permuted_input_ids)
                    .logits[:, target_ids]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )  # [num_classes]
            attributions[idx] = y_pred - y_permute

        attributions = torch.abs(attributions)
        return attributions / torch.norm(attributions, p=1)

    return occlusion_1_callable


def get_attention_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    **kwargs,
) -> Explainer:
    model.eval()

    @torch.no_grad()
    def attention_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(input_ids, output_attentions=True)
            last_layer_attention = output.attentions[-1]
            last_layer_attention_average_heads = last_layer_attention.mean(1).squeeze(0)
            last_layer_attention_average_heads_cls = last_layer_attention_average_heads[
                0
            ]
        return last_layer_attention_average_heads_cls

    return attention_callable


def get_attingrad_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    **kwargs,
) -> Explainer:
    model.eval()

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return output.logits

    if hasattr(model, "roberta"):
        explainer = captum.attr.LayerGradientXActivation(
            predict, model.roberta.embeddings, multiply_by_inputs=True
        )
    elif hasattr(model, "bert"):
        explainer = captum.attr.LayerGradientXActivation(
            predict, model.bert.embeddings, multiply_by_inputs=True
        )

    @torch.no_grad()
    def attingrad_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        gradient_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    target=target,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            gradient_attributions[:, idx] = attributions.detach().cpu()

        gradient_attributions = gradient_attributions

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(input_ids, output_attentions=True)
            last_layer_attention = output.attentions[-1]
            last_layer_attention_average_heads = last_layer_attention.mean(1).squeeze(0)
            last_layer_attention_average_heads_cls = (
                last_layer_attention_average_heads[0].unsqueeze(1).cpu()
            )

        attingrad = gradient_attributions * last_layer_attention_average_heads_cls

        return attingrad / attingrad.sum(0)

    return attingrad_callable


def get_deeplift_callable(
    model_in: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    """Get a callable DeepLift explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable DeepLift explainer
    """

    if hasattr(model_in, "roberta"):
        model = ModelWrapper(model_in)
        explainer = captum.attr.LayerDeepLift(
            model, model.model.roberta.embeddings, multiply_by_inputs=True
        )
    elif hasattr(model_in, "bert"):
        model = ModelWrapper(model_in)
        explainer = captum.attr.LayerDeepLift(
            model, model.model.bert.embeddings, multiply_by_inputs=True
        )

    def deeplift_callable(
        input_ids: torch.Tensor,
        device: str | torch.device,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate DeepLift attributions for each class in target_ids.

        Args:
            inputs (torch.Tensor): Input token ids [batch_size, sequence_length]
            device (str | torch.device): Device to use
            target_ids (torch.Tensor): Target token ids [num_classes]

        Returns:
            torch.Tensor: Attributions [sequence_length, num_classes]
        """
        sequence_length = input_ids.shape[1]
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )

        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    target=target,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return deeplift_callable


def get_gradient_x_input_callable(
    model: torch.nn.Module,
    **kwargs,
) -> Explainer:
    """Get a callable Gradient x Input explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Gradient x Input explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return output.logits

    if hasattr(model, "roberta"):
        explainer = captum.attr.LayerGradientXActivation(
            predict, model.roberta.embeddings, multiply_by_inputs=True
        )
    elif hasattr(model, "bert"):
        explainer = captum.attr.LayerGradientXActivation(
            predict, model.bert.embeddings, multiply_by_inputs=True
        )

    def gradients_x_input_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    target=target,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return gradients_x_input_callable


def get_integrated_gradient_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    batch_size: int = 16,
    **kwargs,
) -> Explainer:
    """Get a callable Integrated Gradients explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): EOS token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return output.logits

    if hasattr(model, "roberta"):
        explainer = captum.attr.LayerIntegratedGradients(
            predict, model.roberta.embeddings, multiply_by_inputs=True
        )
    elif hasattr(model, "bert"):
        explainer = captum.attr.LayerIntegratedGradients(
            predict, model.bert.embeddings, multiply_by_inputs=True
        )

    def integrated_gradients_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)

        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(target_ids):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    target=target,
                    internal_batch_size=batch_size,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions / class_attributions.sum(0)

    return integrated_gradients_callable


def get_kernelshap_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return output.logits

    explainer = captum.attr.KernelShap(predict)

    @torch.no_grad()
    def kernelshap_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )

        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for idx, target in enumerate(target_ids):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    n_samples=sample_ratio * sequence_length,
                    target=target,
                    feature_mask=feature_mask,
                )
                class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        class_attributions = torch.abs(class_attributions)
        return class_attributions / torch.norm(class_attributions, p=1)

    return kernelshap_callable


@torch.no_grad()
def get_lime_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return output.logits

    explainer = captum.attr.Lime(predict)

    @torch.no_grad()
    def lime_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
        )
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for idx, target in enumerate(target_ids):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    n_samples=sample_ratio * sequence_length,
                    target=target,
                )
                class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        class_attributions = torch.abs(class_attributions)
        return class_attributions / torch.norm(class_attributions, p=1)

    return lime_callable


def get_random_baseline_callable(model, **kwargs) -> Explainer:
    def random_baseline_callable(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        num_classes = target_ids.shape[0]
        attributions = torch.abs(torch.randn((sequence_length, num_classes)))
        return attributions / attributions.sum(0, keepdim=True)

    return random_baseline_callable
