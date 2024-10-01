# here you can see the drawing mechanism of the formal problems
# random_generation/generate_random_proofs.py function: draw_svg

# Drawing code 
# random_generation/draw_svg/get_svg.py 

# Other matplot lib friendly plotting cycleGAN/plot_geometry_problems/mat_plot_lib_2_svg.py

# inference script - cycleGAN/inference.py

# train: cycleGAN/train.py

# train (example) data: /is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/svg_fl_long_first_few.csv

# checkpoint: /is/cluster/fast/scratch/pghosh/temp/Meta-Llama-3.1-8B_dec_only_10

#%%
from pathlib import Path
import pandas as pd
from io import StringIO
import torch

from inference import generate_text, load_model_for_inference
from my_utils.generic_utils import apply_start_end_tags, compress_text_forwaiting_and_eot_tokens


checkpoint = Path("/fast/scratch/pghosh/temp/Meta-Llama-3.1-8B_dec_only_10/checkpoints")
data_filename = Path("/fast/scratch/pghosh/dataset/alpha_geo/geometry/svg_fl_long_first_few.csv")
data_filename = Path("/tmp/test1.csv")
#%%
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch

accelerator = Accelerator()
model, tokenizer, wait_id, fl_init_end_toks, nl_init_end_toks = load_model_for_inference(checkpoint_path=checkpoint)
model = model.decoder
#%%
df = pd.read_csv(data_filename, sep=",")
df.head()
#%%
print(df.loc[0, "nl_statement"])
nl_statement = df.loc[0, "nl_statement"]
fl_statement = df.loc[0, "fl_statement"]
nl_statement, fl_statement
#%%
fl_statement_train, nl_statement_train = apply_start_end_tags([fl_statement], [nl_statement], fl_init_end_toks, nl_init_end_toks)
nl_statement_train, fl_statement_train = nl_statement_train[0], fl_statement_train[0]
#%%
fl_ids = tokenizer(fl_statement_train).input_ids
nl_ids = tokenizer(nl_statement_train).input_ids
nl_fl_ids = nl_ids + fl_ids
nl_fl_ids_tensor = torch.tensor(nl_fl_ids, device=model.device).long()
#%%
model.train()
# model.eval()
with torch.no_grad():
    out = model(input_ids=nl_fl_ids_tensor.unsqueeze(0))
#%%
probs = torch.softmax(out.logits[0], dim=-1)
probs_along = torch.gather(probs, 1, nl_fl_ids_tensor.unsqueeze(0)).squeeze(0)
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(probs_along.cpu().numpy(), label="probs along gt")
ax.plot(probs.max(dim=-1).values.cpu().numpy(), label="max pred probs")
ax.vlines(len(nl_ids), 0, 1, colors='r', label="start of formal")
ax.set_xlabel("Token index")
ax.set_ylabel("Probability")
ax.legend()
#%%
_, max_indices = probs.max(dim=-1)
output_decoded = tokenizer.decode(nl_fl_ids[len(nl_ids):])
max_prob_output_decoded = tokenizer.decode(max_indices[len(nl_ids):])
print("Gt output      :", output_decoded)
print()
print("Max prob output:", compress_text_forwaiting_and_eot_tokens(max_prob_output_decoded, wait_token=tokenizer.decode(wait_id), eot_token=tokenizer.eos_token))
#%%
N, R = 5, 4
array = torch.randn(N, R)  # shape (N, R)
indices = torch.randint(0, R, (N, 1))  # shape (N, 1), values between 0 and R-1

# Use gather to select elements
output = torch.gather(array, 1, indices).squeeze(1)
output
#%%

fl_statement_train, nl_statement_train = apply_start_end_tags([fl_statement], [nl_statement], fl_init_end_toks, nl_init_end_toks)
nl_statement_train, fl_statement_train = nl_statement_train[0], fl_statement_train[0]
inputs = tokenizer(
    nl_statement_train + tokenizer.bos_token + fl_init_end_toks[0],
    return_tensors='pt',
).to(model.device)
inputs
out = model(**inputs)
#%%

out.logits.shape
#%%
output = generate_text(model, tokenizer, fl_init_end_toks, nl_init_end_toks, nl_statement, max_length=3000, num_beams=1, do_sample=False, top_k=50, top_p=0.95)
# model = accelerator.prepare(model.decoder)

#%%
output[len(nl_statement)-10:]
#%%
dataset["max_len"] = dataset.map(lambda row: tokenizer(row["nl_statement"] + row["fl_statement"]))
dataset["max_len"].max()

#%%
model
#%%
df = pd.read_csv(data_filename, sep=",")
df.head()
print(df.loc[0, "nl_statement"])
nl_statement = df.loc[0, "nl_statement"]
fl_statement = df.loc[0, "fl_statement"]
nl_ids = tokenizer(nl_statement)["input_ids"]
fl_ids = tokenizer(fl_statement)["input_ids"]
nl_ids + [tokenizer.bos_token_id] + fl_ids
nl_init_end_toks_ids = tokenizer(nl_init_end_toks, return_tensors='pt').input_ids
fl_init_end_toks_ids = tokenizer(fl_init_end_toks, return_tensors='pt').input_ids
nl_init_end_toks_ids
#%%
fl_statement_train, nl_statement_train = apply_start_end_tags([fl_statement], [nl_statement], fl_init_end_toks, nl_init_end_toks)
fl_statement_train, nl_statement_train = fl_statement_train[0], nl_statement_train[0]
#%%
fl_init_end_toks[0]
#%%
train_ids = tokenizer(nl_statement_train)["input_ids"] + tokenizer(fl_statement_train)["input_ids"]
inference_ids = tokenizer(nl_statement_train + tokenizer.bos_token + fl_init_end_toks[0])["input_ids"]
tokenizer.decode(train_ids)
print(train_ids[:len(inference_ids)][-20:])
print(inference_ids[-20:])
#%%
print(fl_statement_train)
print("inf")
print(fl_init_end_toks[0] + fl_statement + fl_init_end_toks[1])
print()
print(nl_statement_train)
print("inf")
print(nl_init_end_toks[0] + nl_statement + nl_init_end_toks[1])
#%%
natural_texts = (nl_init_end_toks[0] + nl_statement + nl_init_end_toks[1] + tokenizer.bos_token
                     + fl_init_end_toks[0] + fl_statement + fl_init_end_toks[1])
inf_input = tokenizer(natural_texts, return_tensors='pt', max_length=2048, truncation=True).input_ids
train_input = nl_ids + fl_ids
print("train_input:", train_input)
print("inf_input  :", inf_input.tolist()[0])
#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
model
#%%

import gradio as gr

def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="SVG input", container=True, scale=7),
    title="Query model",
    description="Query model",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    retry_btn="Retry",
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
# %%
