#%%
from datasets import DatasetDict, Dataset
# for testing
ds = Dataset.from_dict({
    "formal": ["Hello how are you", "I am fine", "What are you doing?"],
    "natural": ["What are you doing?", "I am fine", "Hello how are you"],
    "total_token_lens": [1510, 339, 45],
})
rephrased_dataset = DatasetDict({"train": ds, "validation": ds, "test": ds})


# insert the code below right after this code:
# rephrased_dataset = prepare_data(
#     args.dataset_dir / 'rephrased-nl_fl_dataset_all.jsonl', seed=seed, nrows=args.nrows_rephrased, 
#     colnames={"formal": "fl_statement", "natural": "rephrase"}
# )
print(f"Rephrased dataset before filtering: {rephrased_dataset}")
rephrased_dataset = rephrased_dataset.filter(lambda x: x["total_token_lens"] <= 1500)
print(f"Rephrased dataset after filtering: {rephrased_dataset}")



#%%
import itertools
import json

import numpy as np

from model_preparation import AutoEncoderLLM

#%%
import torch
a = torch.tensor([4.], requires_grad=True)
b = torch.tensor([5.])
c = torch.tensor([6.])
d = torch.tensor([a,b,c])
d.requires_grad

#%%
import torch
aa = torch.arange(20).reshape(4, 5)
indices = torch.tensor([1, 3, 2, 0])
aa[indices:indices+10]
#%%

model = AutoEncoderLLM.from_pretrained("runs/gpt2dec_only/checkpoints")
#%%
model
#%%


def check_keys_in_jsonl(file_path, nrows=None):
    keys = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in itertools.islice(file, nrows):
            content = json.loads(line)
            if keys is None:
                keys = set(content.keys())
            assert keys == set(content.keys()), f"Keys do not match: {keys} != {set(content.keys())}"

filename = "/home/mmordig/reinforcement/alphageometry/cycleGAN/runs/datasets/arithmetic/rephrased-nl_fl_dataset_all.jsonl"
check_keys_in_jsonl(filename)#, nrows=10)

# %%

xx = train_datasets[1]
indices = np.array([391, 117, 330, 275, 396, 380, 325, 95]) - len(train_datasets[0])
xx[[i for i in indices if i >= 0]]

#%%


#%%
from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("gpt2")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
#%%
tokenizer(
    ["Hello how are you"], 
    ["What are you doing right now?"]
)
#%%
from transformers import Trainer
#%%


#%%
?model
#%%
?model.save_pretrained()
# %%
def combine_handling_none(x, new):
    """append, handling None"""
    if x is None:
        return np.array([new])
    return np.concatenate([x, [new]])
xx = None
xx = combine_handling_none(xx, 2.0); xx
xx = combine_handling_none(xx, 2.1); xx

# np.array([2]).shape

#%%
from my_utils.hf_wrapper import debug_on_error

@debug_on_error
def f():
    return 1/0
f()
# %%
