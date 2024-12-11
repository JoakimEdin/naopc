import pytest
import torch
from datasets import Dataset, DatasetDict
from aopc import Aopc
from pytest_lazyfixture import lazy_fixture


@pytest.fixture(scope="module")
def aopc_instance():
    return Aopc(model_id="distilbert-base-uncased")


@pytest.fixture
def sample_input_row():
    return {
        "input_ids": torch.tensor([101, 2009, 2003, 1037, 2204, 2154, 102]),
        "target_label": 1,
        "attributions": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    }


@pytest.fixture
def sample_dataset():
    data = {
        "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
        "target_label": [1],
        "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_dataset_dict(sample_dataset):
    return DatasetDict({"train": sample_dataset, "test": sample_dataset})


@pytest.mark.parametrize(
    "input_data,word_map,normalization",
    [
        (lazy_fixture("sample_input_row"), None, None),
        (lazy_fixture("sample_input_row"), None, "approx"),
        (lazy_fixture("sample_input_row"), None, "exact"),
        (
            lazy_fixture("sample_input_row"),
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            "approx",
        ),
    ],
)
def test_evaluate_row(aopc_instance, input_data, word_map, normalization):
    result = aopc_instance.evaluate(input_data, word_map, normalization)
    assert isinstance(result, tuple)
    assert len(result) == 3


@pytest.mark.parametrize(
    "input_data,word_map,normalization",
    [
        (lazy_fixture("sample_input_row"), None, None),
        (lazy_fixture("sample_dataset"), None, "approx"),
        (lazy_fixture("sample_dataset"), None, "exact"),
        (lazy_fixture("sample_dataset"), [[0, 1, 2, 3, 4, 5, 6]], "approx"),
    ],
)
def test_evaluate_dataset(aopc_instance, input_data, word_map, normalization):
    result = aopc_instance.evaluate(input_data, word_map, normalization)
    assert isinstance(result, Dataset)
    assert "input_ids" not in result.column_names


@pytest.mark.parametrize(
    "input_data,word_map,normalization",
    [
        (lazy_fixture("sample_input_row"), None, None),
        (lazy_fixture("sample_dataset_dict"), None, "approx"),
        (lazy_fixture("sample_dataset_dict"), None, "exact"),
        (
            lazy_fixture("sample_dataset_dict"),
            {"train": [[0, 1, 2, 3, 4, 5, 6]]},
            "approx",
        ),
    ],
)
def test_evaluate_dataset_dict(
    aopc_instance: Aopc, input_data, word_map, normalization
):
    result = aopc_instance.evaluate(input_data, word_map, normalization)
    assert isinstance(result, DatasetDict)
    for split in result:
        assert "input_ids" not in result[split].column_names
