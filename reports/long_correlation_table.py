import matplotlib.pyplot as plt
import pandas as pd

datasets = ["yelp", "sst2", "imdb"]
length = "long"

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
        results_df = pd.read_csv(
            f"results/aopc_scores_{length}/{dataset}_{model_name.split('/')[1]}.csv"
        )
        aopc_approx_limits = pd.read_parquet(
            f"results/aopc_limits_approx/{dataset}_{length}_no_preprocessing_beam_size_5_{model_name.split('/')[1]}.parquet"
        )

        # Set comprehensiveness_approx and sufficiency_approx to float32
        aopc_approx_limits["comprehensiveness"] = aopc_approx_limits["comprehensiveness"].astype("float32")
        aopc_approx_limits["sufficiency"] = aopc_approx_limits["sufficiency"].astype("float32")

        # Set comprehensiveness and sufficiency to float32
        results_df["comprehensiveness"] = results_df["comprehensiveness"].astype("float32")
        results_df["sufficiency"] = results_df["sufficiency"].astype("float32")

        results_df = results_df.merge(
            aopc_approx_limits,
            on="id",
            suffixes=("", "_approx"),
        )
        results_df["dataset"] = dataset

        results_df["model"] = model_dict[model_name]
        dataframe_list.append(results_df)

df = pd.concat(dataframe_list)
df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])

df["approx_normalized_comprehensiveness"] = (df["comprehensiveness"] - df["sufficiency_approx"]) / (
    df["comprehensiveness_approx"] - df["sufficiency_approx"]
)
df["approx_normalized_sufficiency"] = (df["sufficiency"] - df["sufficiency_approx"]) / (
    df["comprehensiveness_approx"] - df["sufficiency_approx"]
)


df.loc[(df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0, "approx_normalized_comprehensiveness"] = 0
df.loc[(df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0, "approx_normalized_sufficiency"] = 0


dataset_correlations_df = pd.DataFrame()
for dataset in datasets:
    df_dataset = df[df["dataset"] == dataset]
    df_dataset = df_dataset[["explanation_method", "model", "approx_normalized_comprehensiveness", "approx_normalized_sufficiency", "comprehensiveness", "sufficiency"]]

    # Set all columns except explanation_method and model to float32
    for col in df_dataset.columns:
        if col not in ["explanation_method", "model"]:
            df_dataset[col] = df_dataset[col].astype("float32")


    comp_model_order_corr = df_dataset.groupby("model")[["comprehensiveness", "approx_normalized_comprehensiveness"]].mean().corr(method="spearman").iloc[0]
    comp_exp_order_corr = df_dataset.groupby("explanation_method")[["comprehensiveness", "approx_normalized_comprehensiveness"]].mean().corr(method="spearman").iloc[0]
    suff_model_order_corr = df_dataset.groupby("model")[["sufficiency", "approx_normalized_sufficiency"]].mean().corr(method="spearman").iloc[0]
    suff_exp_order_corr = df_dataset.groupby("explanation_method")[["sufficiency", "approx_normalized_sufficiency"]].mean().corr(method="spearman").iloc[0]

    model_order_corrs = pd.concat((comp_model_order_corr, suff_model_order_corr), axis=1)[["comprehensiveness", "sufficiency"]]
    exp_order_corrs = pd.concat((comp_exp_order_corr, suff_exp_order_corr), axis=1)[["comprehensiveness", "sufficiency"]]

    model_order_corrs.to_csv(f"results/correlations/{dataset}_long_model_order_corrs.csv", index=False)
    exp_order_corrs.to_csv(f"results/correlations/{dataset}_long_exp_order_corrs.csv", index=False)