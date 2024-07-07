from feature_attribution_methods import get_pertubation_solver_callable
from feature_interactions.interaction_regression import FeatureInteractionLinearRegression
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

num_independent_features = 126
num_or_feature_pairs = 0
num_and_feature_pairs = 0
model = FeatureInteractionLinearRegression(
    num_independent_features=num_independent_features,
    num_or_feature_pairs=num_or_feature_pairs,
    num_and_feature_pairs=num_and_feature_pairs,
)
model.to("cuda")

total_features = num_independent_features + num_or_feature_pairs*2 + num_and_feature_pairs*2

pertubation_solver = get_pertubation_solver_callable(
    model=model,
    baseline_token_id=0,
    descending=True,
    cls_token_id=-100,
    eos_token_id=-100,
    beam_size=50,
    batch_size=1,
    num_top_features_to_consider=20,
    num_random_features_to_consider=20,
    greedy_initialisation=False,
    is_linear_regression=True,
)

input_ids = torch.tensor([1 for _ in range(total_features)]).unsqueeze(0).float().to(device="cuda")
maximal_aopc_bound = model.get_aopc_bound(input_ids).item()

feature_importances = pertubation_solver(input_ids, target_ids=None, device="cuda")
feature_importance_dict = {i: weight for i, weight in enumerate(feature_importances)}
sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

full_output = model.predict(input_ids).item()

deltas = []
for idx, weight in sorted_importances:
    input_ids[0][idx] = 0
    new_output = model.predict(input_ids).item()
    deltas.append(full_output - new_output)

aopc_bound = sum(deltas) / (full_output*len(deltas))
print(f"Computed AOPC bound: {aopc_bound}")
print(f"Maximal AOPC bound: {maximal_aopc_bound}")
