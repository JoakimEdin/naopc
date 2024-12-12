import pytest
import torch
from datasets import Dataset, DatasetDict
from aopc import Aopc, AopcResult


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
    "input_data,word_map,normalization,beam_size",
    [
        (
            {
                "input_ids": torch.tensor([[101, 2009, 2003, 1037, 2204, 2154, 102]]),
                "target_label": 1,
                "attributions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            None,
            None,
            None,
        ),
        (
            {
                "input_ids": torch.tensor([[101, 2009, 2003, 1037, 2204, 2154, 102]]),
                "target_label": 1,
                "attributions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            None,
            "approx",
            None,
        ),
        (
            {
                "input_ids": torch.tensor([[101, 2009, 2003, 1037, 2204, 2154, 102]]),
                "target_label": 1,
                "attributions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            None,
            "exact",
            None,
        ),
        (
            {
                "input_ids": torch.tensor([[101, 2009, 2003, 1037, 2204, 2154, 102]]),
                "target_label": 1,
                "attributions": [0.1, 0.2, 0.3, 0.4],
            },
            {1: [1], 2: [2, 3], 3: [4], 4: [5]},
            "approx",
            None,
        ),
        (
            {
                "input_ids": torch.tensor([[101, 2009, 2003, 1037, 2204, 2154, 102]]),
                "target_label": 1,
                "attributions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            None,
            "approx",
            2,
        ),
    ],
)
def test_evaluate_row(aopc_instance, input_data, word_map, normalization, beam_size):
    result = aopc_instance.evaluate(
        input_data, word_map=word_map, normalization=normalization, beam_size=beam_size
    )
    assert isinstance(result, dict)
    assert AopcResult(**result)


@pytest.mark.parametrize(
    "input_data,word_map,normalization,beam_size",
    [
        (
            Dataset.from_dict(
                {
                    "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                    "target_label": [1],
                    "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                }
            ),
            None,
            None,
            None,
        ),
        (
            Dataset.from_dict(
                {
                    "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                    "target_label": [1],
                    "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                }
            ),
            None,
            "approx",
            None,
        ),
        (
            Dataset.from_dict(
                {
                    "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                    "target_label": [1],
                    "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                }
            ),
            None,
            "exact",
            None,
        ),
        (
            Dataset.from_dict(
                {
                    "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                    "target_label": [1],
                    "attributions": [[0.1, 0.2, 0.3, 0.4]],
                }
            ),
            {1: [1], 2: [2, 3], 3: [4], 4: [5]},
            "approx",
            None,
        ),
        (
            Dataset.from_dict(
                {
                    "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                    "target_label": [1],
                    "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                }
            ),
            None,
            "approx",
            2,
        ),
    ],
)
def test_evaluate_dataset(
    aopc_instance, input_data, word_map, normalization, beam_size
):
    result = aopc_instance.evaluate(
        input_data, word_map=word_map, normalization=normalization, beam_size=beam_size
    )
    assert isinstance(result, Dataset)
    assert AopcResult(**result[0])


@pytest.mark.parametrize(
    "input_data,word_map,normalization,beam_size",
    [
        (
            DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                            "target_label": [1],
                            "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                        }
                    ),
                    "test": Dataset.from_dict(
                        {
                            "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                            "target_label": [1],
                            "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                        }
                    ),
                }
            ),
            None,
            None,
            None,
        ),
        (
            DatasetDict(
                {
                    "train": Dataset.from_dict(
                        {
                            "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                            "target_label": [1],
                            "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                        }
                    ),
                    "test": Dataset.from_dict(
                        {
                            "input_ids": [[101, 2009, 2003, 1037, 2204, 2154, 102]],
                            "target_label": [1],
                            "attributions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
                        }
                    ),
                }
            ),
            None,
            "approx",
            None,
        ),
    ],
)
def test_evaluate_dataset_dict(
    aopc_instance, input_data, word_map, normalization, beam_size
):
    result = aopc_instance.evaluate(
        input_data, word_map=word_map, normalization=normalization, beam_size=beam_size
    )
    assert isinstance(result, DatasetDict)
    for split in result:
        assert AopcResult(**result[split][0])
