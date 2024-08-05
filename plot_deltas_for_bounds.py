import pandas as pd
import matplotlib.pyplot as plt

datasets = ["sst2", "yelp"]
preprocessing_options = ["no_preprocessing", 
                         "feature_attributions_decompx",
                         "feature_attributions_deeplift",
                         "feature_attributions_gradient_x_input",
                         ]

for dataset in datasets:
    for preprocessing in preprocessing_options:
        for beam_size in [1, 5, 10, 20, 50]:

            model_out_files = [
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_bert-base-uncased-SST-2",
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_roberta-base-SST-2",
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_bert-base-uncased-yelp-polarity",
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_roberta-base-finetuned-yelp-polarity",
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_bert-base-uncased-imdb",
                f"{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_roberta-base-imdb",
            ]

            model_type_to_model_name_dict = {
                "bert-base-uncased-SST-2": "BERT SST-2",
                "roberta-base-SST-2": "RoBERTa SST-2",
                "bert-base-uncased-yelp-polarity": "BERT Yelp",
                "roberta-base-finetuned-yelp-polarity": "RoBERTa Yelp",
                "bert-base-uncased-imdb": "BERT IMDB",
                "roberta-base-imdb": "RoBERTa IMDB",

            }

            datas = [pd.read_parquet(
                f"results/aopc_orders_approx/{model_out_file}.parquet"
            ) for model_out_file in model_out_files]
            model_names = [model_type_to_model_name_dict[model_out_file.split("_")[-1]] for model_out_file in model_out_files]


            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            for data, name in zip(datas, model_names):
                data["comprehensiveness_upper_delta"].plot.hist(ax=axs[0], bins=20, alpha=0.5, label=name)
                axs[0].set_title("Upper Delta")
                axs[0].legend()
                data["sufficiency_lower_delta"].plot.hist(ax=axs[1], bins=20, alpha=0.5, label=name)
                axs[1].set_title("Lower Delta")
                axs[1].legend()
            
            # Add figure title
            preprocessing_string = f"plain Beam search with {beam_size} beams" if preprocessing == "no_preprocessing" else f"{' '.join(preprocessing.split("_")[2:]).title()} attributions + Beam search with {str(beam_size)} beams"
            fig.suptitle(f"Delta Distribution for {dataset} and {preprocessing_string}")

            # Save the plot
            plt.savefig(f"results/aopc_deltas/{dataset}_short_{preprocessing}_beam_size_{str(beam_size)}_all_models.png")
            plt.close()
