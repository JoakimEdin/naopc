# Aopc

The `Aopc` package provides a framework for evaluating model faithfulness using the Area Over the Perturbation Curve (AOPC) metric. It supports Hugging Face models and datasets, specifically tailored for sequence label classification tasks.

## Installation

Install the package via pip:

```bash
pip install aopc
```

## Key Features

- **Support for Hugging Face models and datasets**: Utilize pre-trained models and standard datasets seamlessly.
- **AOPC Evaluation**: Calculate AOPC metrics for attributions.
- **Beam Size Suggestion**: Automatically estimate optimal beam sizes for normalized AOPC using our approximation method.

## Quick Start

### Initialize the `Aopc` Class

Start by configuring `Aopc` with a Hugging Face model, such as `prajjwal1/bert-tiny`:

```python
from aopc import Aopc

aopc = Aopc(model_id="prajjwal1/bert-tiny")
```

### Evaluate Dataset

Load your dataset with Hugging Face's `datasets` library and evaluate it with `Aopc`:
> **Note**: If the dataset has not been tokenized `Aopc` will take care of it.
```python
import datasets

# Load dataset
dset = datasets.load_dataset("stanfordnlp/imdb")

# Evaluate dataset without normalization
new_dset = aopc.evaluate(dset)
```
**Note**: `Aopc.evaluate()` allow either a dictionary, datasets.Dataset or datasets.DatasetDict as input.

### Normalized AOPC with Exact Bounds
Estimate 

```python
new_dset = aopc.evaluate(dset, normalization="exact")
```

### Normalized AOPC with Approximated Bounds
Calculate the suggested beam size for normalized AOPC approximation:

```python
# Estimate Beam Size
beam_size = aopc.get_suggested_beam_size(dset)

# Approximate normalization
new_dset = aopc.evaluate_dset(dset, normalization="approx", beam_size=beam_size)
```

## License

This project is licensed under the MIT License.
