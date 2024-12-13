from functools import partial
import typing

import datasets
from loguru import logger
import pydantic
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aopc.methods import (
    calculate_aopc_for_attributions,
    get_bounds,
)

#  A "placeholder" for a type that will be substituted with a specific type when the `evaluate()` is used.
DatasetTypes = typing.TypeVar(
    "DatasetTypes",
    bound=typing.Union[dict[str, typing.Any], datasets.Dataset, datasets.DatasetDict],
)
WordMap: typing.TypeAlias = dict[int, int] | dict[int, list[int]] | None
NormalizationType: typing.TypeAlias = typing.Literal["exact", "approx"] | None
DictStrKey: typing.TypeAlias = dict[str, typing.Any]
TokenId: typing.TypeAlias = int | str | list[str] | None


class InputModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True, arbitrary_types_allowed=True, from_attributes=False
    )
    input_ids: torch.Tensor | list[int] | None = None
    text: str | None = None
    target_label: int
    attributions: torch.Tensor | list[float]

    @pydantic.model_validator(mode="before")
    def validate_model(cls, data: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if "input_ids" not in data and "text" not in data:
            raise ValueError(
                f"Either `text` or `input_ids` must be provided. Got: {data.keys()}"
            )
        return data


class AopcBounds(pydantic.BaseModel):
    lower_bound: float
    upper_bound: float


class AopcResult(pydantic.BaseModel):
    lower_bound: float | None = None
    upper_bound: float | None = None
    desc_aopc: float
    asc_aopc: float
    normalized_desc_aopc: float | None = None
    normalized_asc_aopc: float | None = None
    normalization_type: NormalizationType


class AsDict:
    """A callable that converts a pydantic model to a dict."""

    def __init__(self, fn: typing.Callable[..., pydantic.BaseModel]) -> None:
        self.fn = fn

    def __call__(self, row: DictStrKey) -> DictStrKey:
        """Call the inner functions and dump to dict."""
        try:
            x = InputModel(**row)
        except pydantic.ValidationError as e:
            raise ValueError(f"Error validating input. Expected input keys: {e}.")
        y = self.fn(**x.model_dump())
        return y.model_dump()


class Aopc:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 1024,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.eos_token_id = self._validate_token_id(
            self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        )
        self.mask_token_id: int = self._validate_token_id(
            self.tokenizer.mask_token_id or self.eos_token_id
        )  # type: ignore
        self.pad_token_id: int = self._validate_token_id(
            self.tokenizer.pad_token_id or self.eos_token_id
        )
        self.bos_token_id: int = self._validate_token_id(
            self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        )

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _validate_token_id(token_id: TokenId) -> int:
        if isinstance(token_id, int):
            return token_id
        if isinstance(token_id, str):
            return int(token_id)
        raise ValueError(
            f"Token ID must be an integer, but got: {type(token_id)} with value {token_id}"
        )

    def _get_bounds(
        self,
        input_ids: torch.Tensor | None,
        target_label: int,
        text: str | None,
        *,
        word_map: WordMap = None,
        normalization: typing.Literal["exact", "approx"] = "approx",
        beam_size: int | None = None,
    ) -> AopcBounds:
        input_ids, _ = self._prepare_input(input_ids, text)
        lower, upper = get_bounds(
            input_ids=input_ids,
            target_label=target_label,
            word_map=word_map,
            normalization=normalization,
            beam_size=beam_size,
            model=self.model,
            device=self.device,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            mask_token_id=self.mask_token_id,
        )
        return AopcBounds(lower_bound=lower, upper_bound=upper)

    def _prepare_input(
        self,
        input_ids: torch.Tensor | None,
        text: str | None,
        attributions: torch.Tensor | list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids  # type: ignore
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        if attributions is not None:
            if isinstance(attributions, list):
                attributions = torch.tensor(attributions)
            attributions = attributions.to(self.device)
        return input_ids.to(self.device), attributions  # type: ignore

    def evaluate_row(
        self,
        input_ids: torch.Tensor | None,
        text: str | None,
        target_label: int,
        attributions: torch.Tensor | list[float],
        *,
        word_map: dict[int, int] | dict[int, list[int]] | None = None,
        beam_size: int | None = None,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
    ) -> AopcResult:
        input_ids, attributions = self._prepare_input(input_ids, text, attributions)  # type: ignore
        desc_aopc = calculate_aopc_for_attributions(
            input_ids=input_ids,
            target_label=target_label,
            attributions=attributions,
            word_map=word_map,
            model=self.model,
            device=self.device,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            mask_token_id=self.mask_token_id,
        )
        asc_aopc = calculate_aopc_for_attributions(
            input_ids=input_ids,
            target_label=target_label,
            attributions=attributions,
            word_map=word_map,
            model=self.model,
            device=self.device,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            mask_token_id=self.mask_token_id,
            descending=False,
        )
        if normalization:
            lower, upper = get_bounds(
                input_ids=input_ids,
                target_label=target_label,
                device=self.device,
                word_map=word_map,
                normalization=normalization,
                beam_size=beam_size,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                mask_token_id=self.mask_token_id,
                model=self.model,
            )
            return AopcResult(
                lower_bound=lower,
                upper_bound=upper,
                desc_aopc=desc_aopc,
                asc_aopc=asc_aopc,
                normalized_desc_aopc=(desc_aopc - lower) / (upper - lower),
                normalized_asc_aopc=(asc_aopc - lower) / (upper - lower),
                normalization_type=normalization,
            )
        return AopcResult(
            lower_bound=None,
            upper_bound=None,
            desc_aopc=desc_aopc,
            asc_aopc=asc_aopc,
            normalized_desc_aopc=None,
            normalized_asc_aopc=None,
            normalization_type=None,
        )

    def get_suggested_beam_size(
        self,
        dset: datasets.Dataset,
        beam_sizes: list[int] = [1, 5, 10, 20, 50],
        **kwargs: typing.Any,
    ):
        prev_upper, prev_lower = 0.5, 0.5
        converge_counter = 0
        converged_beam_size = beam_sizes[0]  # Initialize with the first beam size

        for beam_size in beam_sizes:
            map_fn = partial(
                self._get_bounds,
                word_map=None,
                beam_size=beam_size,
                normalization="approx",
            )
            dset_bounds = dset.map(
                AsDict(map_fn),
                desc=f"Estimating AOPC for beam size: {beam_size}",
                **kwargs,
            )
            avg_upper = torch.tensor(dset_bounds["upper_bound"]).mean().item()
            avg_lower = torch.tensor(dset_bounds["lower_bound"]).mean().item()

            if (
                abs(avg_upper - prev_upper) / prev_upper < 0.01
                and abs(avg_lower - prev_lower) / prev_lower < 0.01
            ):
                converge_counter += 1
                converged_beam_size = beam_size
                if converge_counter == 2:  # Converged for two consecutive beam sizes
                    logger.info(
                        f"Beam size search converged at beam size {converged_beam_size}."
                    )
                    logger.info(
                        f"Current upper bound: {avg_upper}, selected, previous upper bound: {prev_upper}"
                    )
                    logger.info(
                        f"Current lower bound: {avg_lower}, selected, previous lower bound: {prev_lower}"
                    )
                    logger.info("Tolerance: 1%")
                    return converged_beam_size
            else:
                converge_counter = 0
                prev_upper, prev_lower = avg_upper, avg_lower

        logger.info(
            f"Beam size search did not converge. Returning the largest beam size: {beam_sizes[-1]}"
        )
        return beam_sizes[-1]

    def evaluate_dset(
        self,
        dset: datasets.Dataset,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
        word_map: WordMap = None,
        beam_size: int | None = 5,
        **kwargs: typing.Any,
    ) -> datasets.Dataset:
        """Translating a dataset."""
        fn = partial(
            self.evaluate_row,
            normalization=normalization,
            beam_size=beam_size,
            word_map=word_map,
        )
        return dset.map(
            AsDict(fn),
            remove_columns=dset.column_names,
            desc="Estimating AOPC...",
            **kwargs,
        )

    def evaluate(
        self,
        data: DatasetTypes,
        normalization: NormalizationType = None,
        word_map: WordMap = None,
        beam_size: int | None = 5,
        map_kwargs: dict | None = None,
    ) -> DatasetTypes:
        """Translate a row, dataset or dataset dict."""
        map_kwargs = map_kwargs or {}
        if isinstance(data, datasets.Dataset):
            return self.evaluate_dset(
                data,
                normalization,
                word_map=word_map,
                beam_size=beam_size,
                **map_kwargs,
            )
        if isinstance(data, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.evaluate_dset(
                        v,
                        normalization=normalization,
                        beam_size=beam_size,
                        word_map=word_map,
                        **map_kwargs,
                    )
                    for k, v in data.items()
                }
            )  # type: ignore
        if isinstance(data, dict):
            fn = partial(
                self.evaluate_row,
                normalization=normalization,
                beam_size=beam_size,
                word_map=word_map,
            )
            return AsDict(fn)(data)

        raise TypeError(f"Cannot evaluate input of type `{type(data)}`.")


if __name__ == "__main__":
    aopc = Aopc("textattack/bert-base-uncased-imdb")
    dset: datasets.Dataset = datasets.load_dataset(
        "csv", data_files="data/sst2_test_short.csv", split="train"
    )  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    dset = dset.map(
        lambda x: {
            "input_ids": tokenizer(x["text"])["input_ids"],
            "target_label": x["label"],
        }
    )
    # Make dummy attributions
    dset = dset.map(
        lambda x: {
            "attributions": torch.rand(len(x["input_ids"])),
        }
    )
    print(dset)
    # beam_size = aopc.get_suggested_beam_size(dset)
    result = aopc.evaluate_dset(dset, normalization="exact")
    # result = aopc.evaluate_dset(dset, beam_size=5, normalization="approx")
    print(result)
