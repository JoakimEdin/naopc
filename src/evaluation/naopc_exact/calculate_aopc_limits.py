import time

import numpy as np
import pandas as pd
import torch
from numba import jit
from numba.typed import Dict

dataset_names = ["yelp", "sst2"]
model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

CLAMP_NEGATIVE_VALUES = False  # clamp negative logits to 0
LOGITS_SCORES = False


@jit()
def get_key(vector: list):
    key = ""
    for i in vector:
        key += str(i)
        key += ","
    return key


@jit()
def permutations(
    vector: list[int],
    min_order: list[int],
    max_order: list[int],
    min_max_lookup: dict[str, float],
    confident_score_dict: dict[str, float],
    step: int = 0,
):
    # if we've gotten to the end, print the permutation
    if step == len(vector):
        key = ""
        score = 0
        for i in range(0, len(vector) - 1):
            key = get_key(np.sort(vector[: i + 1]))
            score += confident_score_dict[key]

        if score > min_max_lookup["max"]:
            min_max_lookup["max"] = score
            max_order = vector

        if score < min_max_lookup["min"]:
            min_max_lookup["min"] = score
            min_order = vector

    # everything to the right of step has not been swapped yet
    for i in range(step, len(vector)):
        # copy the string (store as array)
        vector_copy = vector.copy()

        # swap the current index with the step
        vector_copy[step], vector_copy[i] = vector_copy[i], vector_copy[step]

        # recurse on the portion of the string that has not been swapped yet (now it's index will begin with step + 1)
        min_order, max_order = permutations(
            vector_copy,
            min_order,
            max_order,
            min_max_lookup,
            confident_score_dict,
            step + 1,
        )

    return min_order, max_order


# compile the functions
d = Dict()
min_max_lookup = Dict()
min_max_lookup["min"] = 100.1
min_max_lookup["max"] = 0.0
d[""] = 0.0
d["1,"] = 1.0

vector = np.arange(1, 1)
min_order = vector.copy()
max_order = vector.copy()
min_order, max_order = permutations(vector, min_order, max_order, min_max_lookup, d)

for dataset_name in dataset_names:
    for model_name in model_names:
        upper_limit_list = []
        lower_limit_list = []
        id_list = []
        upper_limit_order_list = []
        lower_limit_order_list = []
        number_of_elements_list = []
        prob_list = []
        masked_input_list = []

        df = pd.read_parquet(
            f"results/permutation_outputs/{dataset_name}_{model_name.split('/')[1]}.parquet"
        )

        if LOGITS_SCORES:
            df["pred"] = df["positive_logit"]
        else:
            df["pred"] = torch.softmax(
                torch.tensor(df[["positive_logit", "negative_logit"]].values), dim=1
            ).numpy()[:, 0]

        full_input_logit = df[df["word_key"] == "[]"][["id", "pred"]].rename(
            {"pred": "full_input_logit"}, axis=1
        )
        df = df.merge(full_input_logit, on="id")
        df["pred_diff"] = df["full_input_logit"] - df["pred"]

        if CLAMP_NEGATIVE_VALUES:
            df["pred_diff"] = df["pred_diff"].clip(0)

        for element_id in df["id"].unique():
            df_id = df[df["id"] == element_id]

            print(df_id)

            d = Dict()
            min_max_lookup = Dict()

            for index, row in df_id.iterrows():
                key = row["word_key"][1:-1].replace(" ", "")
                if len(key) > 0:
                    key += ","

                d[key] = row["pred_diff"]

            min_max_lookup["min"] = 100.1
            min_max_lookup["max"] = 0.0

            number_of_elements = len(
                df_id.iloc[df_id["word_key"].apply(len).argmax()]["word_key"].split()
            )
            vector = np.arange(1, number_of_elements + 1)
            min_order = vector.copy()
            max_order = vector.copy()

            tic = time.time()
            min_order, max_order = permutations(
                vector, min_order, max_order, min_max_lookup, d
            )
            print("Time taken: ", time.time() - tic)

            min_value = min_max_lookup["min"]
            max_value = min_max_lookup["max"]

            all_mask_diff_value = d[get_key(vector)]

            # calculate the best possible upper_limit and lower_limit
            upper_limit = (max_value + all_mask_diff_value) / number_of_elements
            lower_limit = (min_value + all_mask_diff_value) / number_of_elements

            full_input_output = df_id["full_input_logit"].values[0]
            masked_input_output = df_id[df_id["word_key"] == str(vector.tolist())]["pred"].values[0]
            
            # # normalize the scores according to the full input output and the output when all features are masked
            # upper_limit = (upper_limit) / (full_input_output-empty_input_output)
            # lower_limit = (lower_limit) / (full_input_output-empty_input_output)

            print("Upper limit score: ", upper_limit)
            print("Upper limit order: ", max_order)

            print("Lower limit score: ", lower_limit)
            print("Lower limit order: ", min_order)

            id_list.append(element_id)
            upper_limit_list.append(upper_limit)
            lower_limit_list.append(lower_limit)
            upper_limit_order_list.append(max_order)
            lower_limit_order_list.append(min_order)
            number_of_elements_list.append(number_of_elements)
            prob_list.append(full_input_output)
            masked_input_list.append(masked_input_output)


        results_df = pd.DataFrame(
            {
                "id": id_list,
                "upper_limit": upper_limit_list,
                "lower_limit": lower_limit_list,
                "upper_limit_order": upper_limit_order_list,
                "lower_limit_order": lower_limit_order_list,
                "number_of_elements": number_of_elements_list,
                "prob": prob_list,
                "masked_input": masked_input_list,
            }
        )
        results_df.to_parquet(
            f"results/aopc_limits_exact/{dataset_name}_{model_name.split('/')[1]}.parquet"
        )
