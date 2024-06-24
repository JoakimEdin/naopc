import torch
from datasets import Dataset
import torch.utils
import itertools

class PerturbDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Dataset, mask_token_id: int, pad_token_id: int, input_ids_column_name: str = "input_ids"):
        self.dataset = dataset
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.input_ids_column_name = input_ids_column_name


    def __iter__(self) -> tuple[int, str, torch.Tensor]:
        for idx, example in enumerate(self.dataset):
            input_ids = torch.tensor(example[self.input_ids_column_name])
            permutation = list(range(1, len(input_ids) - 1))
            for i in range(len(permutation) + 1):
                for mask_indices in itertools.combinations(permutation, i):
                    temp = input_ids.clone()
                    if len(mask_indices) > 0:
                        temp[list(mask_indices)] = self.mask_token_id
                    yield idx, str(sorted(mask_indices)), temp
    
    def collate_fn(self, batch):
        ids, key, input_ids = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).float()
        return list(ids), (key), input_ids, attention_mask



        

