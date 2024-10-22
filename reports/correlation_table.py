import pandas as pd

datasets = ["yelp", "sst2", "imdb"]
lengths = ["long", "short"]

model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

model_dict = {
    "textattack/roberta-base-SST-2": "RoBERTa$_{\\text{SST2}}$",
    "textattack/bert-base-uncased-SST-2": "BERT$_{\\text{SST2}}$",
    "textattack/bert-base-uncased-yelp-polarity": "BERT$_{\\text{Yelp}}$",
    "VictorSanh/roberta-base-finetuned-yelp-polarity": "RoBERTa$_{\\text{Yelp}}$",
    "textattack/bert-base-uncased-imdb": "BERT$_{\\text{IMDB}}$",
    "textattack/roberta-base-imdb": "RoBERTa$_{\\text{IMDB}}$",
}

explainer_dict = {
    "occlusion_1": "Occlusion@1",
    "lime": "LIME",
    "kernelshap": "SHAP",
    "integrated_gradient": "IG",
    "random_baseline": "Random",
    "gradient_x_input": "GxI",
    "attention": "Attention",
    "attingrad": "AttInGrad",
    "decompx": "DecompX",
    "deeplift": "DeepLIFT",
}
dataframe_list = []
for model_name in model_names:
    for dataset in datasets:
        for length in lengths:
            if dataset == "imdb" and length == "short":
                continue
            results_df = pd.read_csv(
                f"results/aopc_scores_{length}/{dataset}_{model_name.split('/')[1]}.csv"
            )
            aopc_approx_limits = pd.read_parquet(
                f"results/aopc_limits_approx/{dataset}_{length}_no_preprocessing_beam_size_5_{model_name.split('/')[1]}.parquet"
            )

            # Set comprehensiveness_approx and sufficiency_approx to float32
            aopc_approx_limits["comprehensiveness"] = aopc_approx_limits[
                "comprehensiveness"
            ].astype("float32")
            aopc_approx_limits["sufficiency"] = aopc_approx_limits[
                "sufficiency"
            ].astype("float32")

            # Set comprehensiveness and sufficiency to float32
            results_df["comprehensiveness"] = results_df["comprehensiveness"].astype(
                "float32"
            )
            results_df["sufficiency"] = results_df["sufficiency"].astype("float32")

            results_df = results_df.merge(
                aopc_approx_limits,
                on="id",
                suffixes=("", "_approx"),
            )
            results_df["dataset"] = dataset + "_" + length

            results_df["model"] = model_dict[model_name]
            dataframe_list.append(results_df)

df = pd.concat(dataframe_list)
df = df[~df["explanation_method"].isin(["AttInGrad", "Random"])]
df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])

df["approx_normalized_comprehensiveness"] = (
    df["comprehensiveness"] - df["sufficiency_approx"]
) / (df["comprehensiveness_approx"] - df["sufficiency_approx"])
df["approx_normalized_sufficiency"] = (df["sufficiency"] - df["sufficiency_approx"]) / (
    df["comprehensiveness_approx"] - df["sufficiency_approx"]
)


df.loc[
    (df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0,
    "approx_normalized_comprehensiveness",
] = 0
df.loc[
    (df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0,
    "approx_normalized_sufficiency",
] = 0


results = []
for dataset in df["dataset"].unique():
    df_dataset = df[df["dataset"] == dataset]
    df_dataset = df_dataset[
        [
            "explanation_method",
            "model",
            "approx_normalized_comprehensiveness",
            "approx_normalized_sufficiency",
            "comprehensiveness",
            "sufficiency",
        ]
    ]

    # Set all columns except explanation_method and model to float32
    for col in df_dataset.columns:
        if col not in ["explanation_method", "model"]:
            df_dataset[col] = df_dataset[col].astype("float32")

    # calculate the mean scores
    df_dataset_mean = df_dataset.groupby(["model", "explanation_method"])[
        [
            "comprehensiveness",
            "approx_normalized_comprehensiveness",
            "sufficiency",
            "approx_normalized_sufficiency",
        ]
    ].mean()

    comp_exp_order_corr = (
        df_dataset_mean.groupby("model")[
            ["comprehensiveness", "approx_normalized_comprehensiveness"]
        ]
        .apply(
            lambda x: x["comprehensiveness"].corr(
                x["approx_normalized_comprehensiveness"], method="kendall"
            )
        )
        .mean()
    )
    comp_model_order_corr = (
        df_dataset_mean.groupby("explanation_method")[
            ["comprehensiveness", "approx_normalized_comprehensiveness"]
        ]
        .apply(
            lambda x: x["comprehensiveness"].corr(
                x["approx_normalized_comprehensiveness"], method="kendall"
            )
        )
        .mean()
    )
    suff_exp_order_corr = (
        df_dataset_mean.groupby("model")[
            ["sufficiency", "approx_normalized_sufficiency"]
        ]
        .apply(
            lambda x: x["sufficiency"].corr(
                x["approx_normalized_sufficiency"], method="kendall"
            )
        )
        .mean()
    )
    suff_model_order_corr = (
        df_dataset_mean.groupby("explanation_method")[
            ["sufficiency", "approx_normalized_sufficiency"]
        ]
        .apply(
            lambda x: x["sufficiency"].corr(
                x["approx_normalized_sufficiency"], method="kendall"
            )
        )
        .mean()
    )
    results.append((dataset, "Model", comp_model_order_corr,suff_model_order_corr))
    results.append((dataset, "FA", comp_exp_order_corr, suff_exp_order_corr))
   
df_results = pd.DataFrame(results, columns=["Dataset", "Group", "Comprehensiveness", "Sufficiency"])
print(df_results.to_latex(index=False, float_format="%.2f"))
