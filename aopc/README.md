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

# Load a tokenizer and generate some random attributions
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
dset = dset.map(
    lambda x: {"input_ids": aopc.tokenizer(x["text"], truncation=True)["input_ids"]}
)
dset = dset.map(
    lambda x: {"attributions": torch.rand(len(x["input_ids"]))}
)

# Evaluate dataset without normalization
new_dset = aopc.evaluate(dset)
```
**Note**: `Aopc.evaluate()` allow either a `dict()`, `datasets.Dataset` or `datasets.DatasetDict` as input.

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

### Word map usage

For some use-cases we might be interested in measuring faithfulness on attributions that are on a word level (or some other combination of tokens) while the tokenization is on the subword level. For this we support having a word map per row. A word map is a mapping from word index to list of token indices. An example:

```Python

tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-ag-news")
text = "Truly horrendous"
input_ids = tokenizer(text)["input_ids"]
> [0, 565, 26582, 29577, 2]

# A word map would map "Truly" to the tokens 565 (T) and 26582 (ruly), and "horrendous" to 29577.  
word_map = {0: [0], 1: [1, 2], 2: [3], 3: [4]}

aopc = Aopc("textattack/roberta-base-ag-news")
aopc.evaluate_row(input_ids=input_ids, target_label=1, word_map=word_map, attributions=torch.rand(len(input_ids)))
```

## License 

This project is licensed under the MIT License.

## Citation

### Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability

```
@misc{edin2024normalizedaopcfixingmisleading,
      title={Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability}, 
      author={Joakim Edin and Andreas Geert Motzfeldt and Casper L. Christensen and Tuukka Ruotsalo and Lars Maaløe and Maria Maistro},
      year={2024},
      eprint={2408.08137},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.08137}, 
}
```

