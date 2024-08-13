🌟 Official repository for the paper [Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](https://arxiv.org/)

## Abstract
> Deep neural network predictions are notoriously difficult to interpret. Feature attribution methods aim to explain these predictions by identifying the contribution of each input feature. Faithfulness, often evaluated using the area over the perturbation curve (AOPC), reflects feature attributions' accuracy in describing the internal mechanisms of deep neural networks. However, many studies rely on AOPC for comparing faithfulness across different models, which we show can lead to false conclusions about models' faithfulness. Specifically, we find that AOPC is sensitive to variations in the model, resulting in unreliable cross-model comparisons. Moreover, AOPC scores are difficult to interpret in isolation without knowing the model-specific lower and upper limits. To address these issues, we propose a normalization approach, Normalized AOPC (NAOPC), enabling consistent cross-model evaluations and more meaningful interpretation of individual scores. Our experiments demonstrate that this normalization can radically change AOPC results, questioning the conclusions of earlier studies and offering a more robust framework for assessing feature attribution faithfulness.


## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Citation](#citation)

## Setup
<details>
<summary>Install poetry</summary>

Detailed steps to get the development environment up and running.

### Install Poetry

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

</details>

<details>
<summary>Install dependencies</summary>
Clone the repository and navigate to the project directory

```bash
git clone https://github.com/JoakimEdin/faithfulness.git
cd faithfulness
```
Then install dependencies
```bash
make install
```

</details>

## Usage



## Citation

```

```
