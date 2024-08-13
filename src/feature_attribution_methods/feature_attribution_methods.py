from typing import Callable, Optional

import captum
import torch

from src.feature_attribution_methods.decompx.decompx_utils import DecompXConfig

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
