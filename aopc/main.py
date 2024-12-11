from functools import partial
import typing

import datasets
from loguru import logger
import pydantic
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aopc.methods import (
    calculate_aopc_for_attributions,
    calculate_aopc_bounds,
    calculate_aopc_bounds_exact,
)

#  A "placeholder" for a type that will be substituted with a specific type when the `evaluate()` is used.
DatasetTypes = typing.TypeVar(
    "DatasetTypes",
    bound=typing.Union[dict[str, typing.Any], datasets.Dataset, datasets.DatasetDict],
)
WordMap: typing.TypeAlias = torch.Tensor | list[int] | None
NormalizationType: typing.TypeAlias = typing.Literal["exact", "approx"] | None


class InputModel(pydantic.BaseModel):
    input_ids: torch.Tensor
    target_label: int
    attributions: torch.Tensor | list[float]
    word_map: WordMap = None
    normalization: NormalizationType = "approx"


class Aopc:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 1024,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.baseline_token_id = self.tokenizer.mask_token_id
        if self.tokenizer.mask_token_id is None:
            self.baseline_token_id = self.tokenizer.eos_token_id
            logger.warning(
                "Tokenizer does not have a MASK token. Using EOS token as MASK token."
            )

        self.cls_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    @staticmethod
    def _calculate_aopc(
        input_ids: torch.Tensor,
        target_label: int,
        attributions: torch.Tensor | list[float],
        word_map: torch.Tensor | list[int] | None = None,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
    ) -> tuple[float, float, float]:
        if normalization == "approx":
            return calculate_aopc_bounds(input_ids, target_label, attributions)
        if normalization == "exact":
            return calculate_aopc_bounds_exact(input_ids, target_label, attributions)
        return calculate_aopc_for_attributions(input_ids, target_label, attributions)

    @staticmethod
    def evaluate_row(
        row: dict[str, typing.Any],
        word_map: torch.Tensor | list[int] | None = None,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
    ) -> tuple[float, float, float]:
        try:
            x = InputModel(**row)
        except pydantic.ValidationError:
            raise ValueError(
                f"Error validating input. Expected input keys: {InputModel.model_fields}"
            )
        return Aopc._calculate_aopc(
            input_ids=x.input_ids,
            attributions=x.attributions,
            target_label=x.target_label,
            word_map=word_map,
            normalization=normalization,
        )

    def evaluate_dset(
        self,
        dset: datasets.Dataset,
        word_map: torch.Tensor | list[int] | None = None,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
        **kwargs: typing.Any,
    ) -> datasets.Dataset:
        """Translating a dataset."""
        fn = partial(self.evaluate_row, word_map=word_map, normalization=normalization)
        return dset.map(
            fn, remove_columns=dset.column_names, desc="Estimating AOPC...", **kwargs
        )

    def evaluate(
        self,
        x: DatasetTypes,
        word_map: WordMap,
        normalization: NormalizationType,
        map_kwargs: dict | None = None,
    ) -> DatasetTypes:
        """Translate a row, dataset or dataset dict."""
        map_kwargs = map_kwargs or {}
        if isinstance(x, datasets.Dataset):
            return self.evaluate_dset(x, word_map, normalization, **map_kwargs)
        if isinstance(x, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.evaluate_dset(v, word_map, normalization, **map_kwargs)
                    for k, v in x.items()
                }
            )  # type: ignore
        if isinstance(x, dict):
            return self.evaluate_row(x, word_map, normalization)

        raise TypeError(f"Cannot evaluate input of type `{type(x)}`")
