#%%

# better processing using hf datasets

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset


def detect_hf_dataset_load_type(filename) -> str:
    if str(filename).endswith('.csv'):
        return 'csv'
    elif str(filename).endswith('.jsonl') or str(filename).endswith('.json'):
        return 'json'
    else:
        raise ValueError(f"cannot handle file type for filename {filename}")
    
def prepare_data(filename, test_size=0.1, seed=None, colnames=None, **kwargs) -> DatasetDict:
    """
    Given a data path, return a hf dataset (which can also be used like torch.Dataset)
    
    Args:
        kwargs: additional arguments to pass to load_dataset
        colnames: maps formal and natural column names to the actual column names in the dataset
    """
    load_type = detect_hf_dataset_load_type(filename)
    nrows = None
    if load_type == "json":
        if "nrows" in kwargs:
            nrows = kwargs.pop("nrows")
    dataset = load_dataset(load_type, data_files=[str(filename)], split='train', **kwargs)
    if nrows is not None:
        dataset = dataset.select(min(len(dataset), range(nrows)))
    print(f"Dataset loaded, of size {len(dataset)}")
    if colnames is None:
        colnames = {"formal": "fl_statement", "natural": "nl_statement"}
    dataset = dataset.rename_columns(({colnames["formal"]: "formal", colnames["natural"]: "natural"}))
    if 'total_token_lens' in colnames:
        dataset = dataset.select_columns(["formal", "natural", "total_token_lens"])
    else:
        dataset = dataset.select_columns(["formal", "natural"])
    dataset = dataset.with_format("torch")

    generator = np.random.default_rng(seed)
    assert 0 <= 2 * test_size <= 1, f"got {test_size}"
    temp_ds = dataset.train_test_split(test_size=0.1 * 2, shuffle=True, generator=generator)
    train_ds = temp_ds["train"]
    temp_ds2 = temp_ds["test"].train_test_split(test_size=0.5, shuffle=True, generator=generator)
    val_ds = temp_ds2["train"]
    test_ds = temp_ds2["test"]
    
    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

import torch
class AlwaysSameElementDataset(torch.utils.data.Dataset):
    """Always return the same element from the dataset"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.dataset[0]

class CombinedDataset(torch.utils.data.Dataset):
    """Combine two datasets"""
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    
    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
    
    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]
        
class MixedDatasetSampler(torch.utils.data.WeightedRandomSampler):
    """
    Sample from two datasets with a certain ratio, use with CombineDatasets

    Args:
        dataset_lens: tuple of lengths of the two datasets
        ratio: ratio of samples dataset2/dataset1, not the sampling weight of samples in dataset2 (but scaled by dataset length)
    """

    def __init__(
        self, dataset_lens, ratio=1.0, num_samples=None
    ):
        assert len(dataset_lens) == 2
        
        print(f"Creating MixedDatasetSampler with lens {dataset_lens} and ratio {ratio}")
        
        # threshold = 1/(1 + ratio) # ratio=2:1 -> threshold=1/3=prob to sample from dataset1
        ratio *= dataset_lens[0] / dataset_lens[1]
        weights = torch.tensor([1] * dataset_lens[0] + [ratio] * dataset_lens[1], dtype=torch.double)
        if num_samples is None:
            num_samples = dataset_lens[0]
        super().__init__(weights=weights, num_samples=num_samples, replacement=True)
        
        self.epoch = 0
        self.initial_seed = torch.random.initial_seed()  # for cpu only

    def _set_generator_for_epoch(self):
        if self.generator is None:
            self.generator = torch.Generator()

        # Allow `self.epoch` to modify the seed of the generator
        seed = self.epoch + self.initial_seed
        # print("Setting seed at epoch", self.epoch, seed)
        self.generator.manual_seed(seed)

    def set_epoch(self, epoch: int):
        "Sets the current iteration of the sampler."
        self.epoch = epoch

    def __iter__(self):
        self._set_generator_for_epoch()
        yield from super().__iter__()
        self.set_epoch(self.epoch + 1)


if __name__ == "__main__":
    import re
    # filename = "runs/datasets/arithmetic/nl_fl.csv"
    # seed = 42
    # test_size = 0.1
    #
    # ds = prepare_data(filename, test_size, seed)
    # print(ds)
    # print(ds["train"][0])
    #
    #
    # ds1 = Dataset.from_dict({"a": list(range(-20, 0))})
    # ds2 = Dataset.from_dict({"a": list(range(1, 3))})
    # combined_ds = CombinedDataset(ds1, ds2)
    # # combined_ds
    # # sampler = MixedDatasetSampler((len(ds1), len(ds2)), ratio=1.0)
    # len_ds1 = 20
    # ratio = 2.0
    # sampler = MixedDatasetSampler((len_ds1, 3), ratio=ratio, num_samples=1000)
    # indices = list(sampler)
    # fraction = len([idx for idx in indices if idx >= len_ds1]) / len(indices)
    # print(f"Fraction {fraction}, expected {ratio/(1+ratio):.2f}")
    #

    ds = Dataset.from_dict({
        "nl_statement": ["Hello how are you", "I am \n \r fine", "What are you doing?"],
        "fl_statement": ["What are you doing?", "I am fine", "Hello how are you"],
        "rephrase": ["What are you doing?", "I am fine", "Hello how are you"],
        "total_token_lens": [1510, 339, 45],
    })

    colnames = {"formal": "fl_statement", "natural": "rephrase", "total_token_lens": "total_token_lens"}

    dataset = ds.rename_columns(({colnames["formal"]: "formal", colnames["natural"]: "natural"}))
    if 'total_token_lens' in colnames:
        dataset = dataset.select_columns(["formal", "natural", "total_token_lens"])
    else:
        dataset = dataset.select_columns(["formal", "natural"])
    dataset = dataset.with_format("torch")

    rephrased_dataset = DatasetDict({"train": dataset, "validation": dataset, "test": dataset})
    print(f"Rephrased dataset before filtering: {rephrased_dataset}")
    rephrased_dataset = rephrased_dataset.filter(lambda x: x["total_token_lens"] <= 1500)
    rephrased_dataset = rephrased_dataset.map(
        lambda x: {"natural": [re.sub(r'\r\n|\r|\n', '', text) for text in x["natural"]]},
        batched=True,
    )
    print(f"Rephrased dataset after filtering: {rephrased_dataset}")
# %%
