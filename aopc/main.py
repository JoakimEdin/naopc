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
WordMap: typing.TypeAlias = dict[int, list[int]]
NormalizationType: typing.TypeAlias = typing.Literal["exact", "approx"] | None


class InputModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True, from_attributes=False)
    input_ids: torch.Tensor | list[int] | None = None
    text: str | None = None
    target_label: int
    attributions: torch.Tensor | list[float]
    word_map: WordMap = None
    normalization: NormalizationType = "approx"
    beam_size: int | None = None

    @pydantic.model_validator(mode="before")
    def validate_model(cls, v):
        text_present = "text" in v
        input_ids_present = "input_ids" in v
        if not text_present and not input_ids_present:
            raise ValueError("Either `text` or `input_ids` must be provided.")

    

class AOPCResult(pydantic.BaseModel):
    lower_bound: float
    upper_bound: float
    desc_aopc: float
    asc_aopc: float
    normalized_desc_aopc: float
    normalized_asc_aopc: float
    normalization_type: NormalizationType

class Aopc:
    def __init__(
        self,
        model_id: str,
        batch_size: int = 1024,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id or self.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model.eval()
        self.model.to(self.device)

    def _get_bounds(
            self,
            input_ids: torch.Tensor | None,
            text: str | None,
            target_label: int,
            word_map: torch.Tensor | list[int] | None = None,
            normalization: typing.Literal["exact", "approx"] | None = "approx",
            beam_size: int | None = None,
    ) -> tuple[float, float]:
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
        return {"lower_bound": lower, "upper_bound": upper}
    
    def _prepare_input(self, input_ids: torch.Tensor | None, text: str | None, attributions: torch.Tensor | list[float] | None = None) -> torch.Tensor:
        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        if attributions:
            if isinstance(attributions, list):
                attributions = torch.tensor(attributions)
        return input_ids.to(self.device), attributions

    def _calculate_aopc(
        self,
        input_ids: torch.Tensor | None,
        text: str | None,
        target_label: int,
        attributions: torch.Tensor | list[float],
        word_map: torch.Tensor | list[int] | None = None,
        beam_size: int | None = None,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
    ) -> tuple[float, float, float]:
        input_ids, attributions = self._prepare_input(input_ids, text, attributions)
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
        return AOPCResult(
            lower_bound=lower,
            upper_bound=upper,
            desc_aopc=desc_aopc,
            asc_aopc=asc_aopc,
            normalized_desc_aopc=(desc_aopc - lower) / (upper - lower),
            normalized_asc_aopc=(asc_aopc - lower) / (upper - lower),
            normalization_type=normalization,
        )
        
    
    def get_bounds_for_row(
            self,
            row: dict[str, typing.Any],
            beam_size: int | None = None,
            normalization: typing.Literal["exact", "approx"] | None = "approx",
    ) -> tuple[float, float]:
        try:
            x = {**row, "beam_size": beam_size, "normalization": normalization}
            #x = InputModel(**row, beam_size=beam_size, normalization=normalization)
        except pydantic.ValidationError as e:
            print(e)
            raise ValueError(
                f"Error validating input. Expected input keys: {InputModel.model_fields} but got {list(row.keys())}"
            )
        return self._get_bounds(
            input_ids=x["input_ids"],
            text=x["text"],
            target_label=x["target_label"],
            word_map=x.get("word_map"),
            normalization=x["normalization"],
            beam_size=x["beam_size"],
        )
        

    def evaluate_row(
        self,
        row: dict[str, typing.Any],
        normalization: typing.Literal["exact", "approx"] | None = "approx",
        beam_size: int | None = None,
    ) -> dict[str, float]:
        try:
            x = {**row, "beam_size": beam_size, "normalization": normalization}
            #x = InputModel(**row)
        except pydantic.ValidationError as e:
            print(e)
            raise ValueError(
                f"Error validating input. Expected input keys: {InputModel.model_fields}"
            )
        return self._calculate_aopc(
            input_ids=x["input_ids"],
            text=x["text"],
            target_label=x["target_label"],
            attributions=x["attributions"],
            word_map=x.get("word_map"),
            normalization=x["normalization"],
            beam_size=x["beam_size"],
        ).model_dump()
    
    def get_suggested_beam_size(
            self,
            dset: datasets.Dataset,
            beam_sizes: list[int] = [1, 5, 10, 20, 50],
            **kwargs: typing.Any,
    ):
        prev_upper, prev_lower = 0.5, 0.5
        converge_counter = 0
        for beam_size in beam_sizes:
            map_fn = partial(self.get_bounds_for_row, beam_size=beam_size, normalization="approx")
            dset_bounds = dset.map(
                map_fn, desc=f"Estimating AOPC for beam size: {beam_size}", **kwargs
            )
            avg_upper = torch.tensor(dset_bounds["upper_bound"]).mean().item()
            avg_lower = torch.tensor(dset_bounds["lower_bound"]).mean().item()
            if abs(avg_upper - prev_upper) / prev_upper < 0.01 and abs(avg_lower - prev_lower) / prev_lower < 0.01:
                if converge_counter == 1:
                    logger.info(f"Beam size search converged at beam size {converged_beam_size}.")
                    logger.info(f"Current upper bound: {avg_upper}, selected, previous upper bound: {prev_upper}")
                    logger.info(f"Current lower bound: {avg_lower}, selected, previous lower bound: {prev_lower}")
                    logger.info(f"Tolerance: 1%")
                    return converged_beam_size
                else:
                    converged_beam_size = beam_size
                    converge_counter += 1
            else:
                converge_counter = 0
                prev_upper, prev_lower = avg_upper, avg_lower
        logger.info(f"Beam size search did not converge. Current upper bound: {avg_upper}, previous upper bound: {prev_upper}")
        return beam_sizes[-1]
        

    def evaluate_dset(
        self,
        dset: datasets.Dataset,
        normalization: typing.Literal["exact", "approx"] | None = "approx",
        beam_size: int | None = 5,
        **kwargs: typing.Any,
    ) -> datasets.Dataset:
        """Translating a dataset."""
        fn = partial(self.evaluate_row, normalization=normalization, beam_size=beam_size)
        return dset.map(
            fn, remove_columns=dset.column_names, desc="Estimating AOPC...", **kwargs
        )

    def evaluate(
        self,
        data: DatasetTypes,
        normalization: NormalizationType,
        beam_size: int | None = 5,
        map_kwargs: dict | None = None,
    ) -> DatasetTypes:
        """Translate a row, dataset or dataset dict."""
        map_kwargs = map_kwargs or {}
        if isinstance(data, datasets.Dataset):
            return self.evaluate_dset(data, normalization, beam_size, **map_kwargs)
        if isinstance(data, datasets.DatasetDict):
            return datasets.DatasetDict(
                {
                    k: self.evaluate_dset(v, normalization, beam_size, **map_kwargs)
                    for k, v in data.items()
                }
            )  # type: ignore
        if isinstance(data, dict):
            return self.evaluate_row(data, normalization, beam_size)

        raise TypeError(f"Cannot evaluate input of type `{type(x)}`")


if __name__ == "__main__":
    aopc = Aopc("textattack/bert-base-uncased-imdb")        
    dset = datasets.load_dataset(
            "csv", data_files=f"data/sst2_test_short.csv", split="train"
        )
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
    #beam_size = aopc.get_suggested_beam_size(dset)
    result = aopc.evaluate_dset(dset, normalization="exact")
    #result = aopc.evaluate_dset(dset, beam_size=5, normalization="approx")
    print(result)