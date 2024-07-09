import random
import torch
import numpy as np

class FeatureInteractionLinearRegression:

    def __init__(self, num_independent_features, num_or_feature_pairs, num_and_feature_pairs, w_lower_bound=1, w_upper_bound=10, w_exp=1):
        self.num_independent_features = num_independent_features
        self.num_or_feature_pairs = num_or_feature_pairs
        self.total_or_features = num_or_feature_pairs * 2
        self.num_and_feature_pairs = num_and_feature_pairs
        self.total_and_features = num_and_feature_pairs * 2

        self.independent_start_index = 0
        self.or_start_index = num_independent_features
        self.and_start_index = num_independent_features + self.total_or_features

        self.independent_feature_weights = torch.tensor([random.uniform(w_lower_bound, w_upper_bound)**w_exp for _ in range(num_independent_features)], dtype=torch.float32)
        self.or_feature_pair_weights = torch.tensor([random.uniform(w_lower_bound, w_upper_bound)**w_exp for _ in range(num_or_feature_pairs)], dtype=torch.float32)
        self.and_feature_pair_weights = torch.tensor([random.uniform(w_lower_bound, w_upper_bound)**w_exp for _ in range(num_and_feature_pairs)], dtype=torch.float32)

    def get_feature_importances(self, descending=True):
        independent_feature_importances = {i: weight.cpu() for i, weight in zip(range(self.num_independent_features), self.independent_feature_weights)}
        or_feature_importances = {}

        for i, weight in enumerate(self.or_feature_pair_weights):
            or_feature_importances[(i*2)+self.or_start_index] = (weight/2).cpu()
            or_feature_importances[(i*2+1)+self.or_start_index] = (weight/2).cpu()
        
        and_feature_importances = {}
        for i, weight in enumerate(self.and_feature_pair_weights):
            and_feature_importances[(i*2)+self.and_start_index] = weight.cpu()
            and_feature_importances[(i*2+1)+self.and_start_index] = 0

        all_importances = {**independent_feature_importances, **or_feature_importances, **and_feature_importances}
        sorted_importances = sorted(all_importances.items(), key=lambda x: x[1], reverse=descending)
        return sorted_importances
    
    def get_aopc_bound(self, x, descending=True):
        full_output = self.predict(x).cpu()
        sorted_importances = self.get_feature_importances(descending=descending)
        deltas = [full_output - (full_output - weight) for _, weight in sorted_importances]
        deltas = np.cumsum(deltas)
        aopc_bound = sum(deltas) / (full_output*len(deltas))
        return aopc_bound

    @torch.no_grad()
    def predict(self, X):
        independent_features = X[:, :self.num_independent_features]
        or_features = X[:, self.num_independent_features:self.num_independent_features+self.total_or_features]
        and_features = X[:, self.num_independent_features+self.total_or_features:]

        out_sum = 0

        # Calculate sum for independent features
        out_sum += torch.matmul(independent_features, self.independent_feature_weights)

        # Calculate sum for OR features
        if or_features.shape[1] % 2 == 0 and or_features.shape[1] > 0:
            or_features = or_features.view(or_features.size(0), -1, 2)
            x_or_val = torch.max(or_features, dim=2).values
            out_sum += torch.matmul(x_or_val, self.or_feature_pair_weights)

        # Calculate sum for AND features
        if and_features.shape[1] % 2 == 0 and and_features.shape[1] > 0:
            and_features = and_features.view(and_features.size(0), -1, 2)
            x_and_val = torch.min(and_features, dim=2).values
            out_sum += torch.matmul(x_and_val, self.and_feature_pair_weights)

        return out_sum
    
    def __call__(self, x):
        return self.predict(x)
    

    def to(self, device):
        self.independent_feature_weights = self.independent_feature_weights.to(device)
        self.or_feature_pair_weights = self.or_feature_pair_weights.to(device)
        self.and_feature_pair_weights = self.and_feature_pair_weights.to(device)