#%%
import itertools
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import pandas as pd
import json
import os
import accelerate

from accelerate import PartialState, Accelerator

from model_preparation import AutoEncoderLLM
from my_utils.generic_utils import numpify
try:
    from .generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe, \
        CustomJSONserializer, apply_start_end_tags
except ImportError:
    from my_utils.generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe, \
        CustomJSONserializer, apply_start_end_tags


class Checkpointer:
    """Save model, but only if it is better than the previous best model"""
    def __init__(self, output_dir, args):
        self.prev_validation_loss = 9999999999
        self.output_dir = output_dir
        self.args_dict = vars(args)

    def checkpoint(self, accelerator, model, validation_loss):
        if accelerator.is_main_process:
            if self.args_dict is not None:
                with open(os.path.join(self.output_dir, 'cmd_args.json'), 'w') as json_file:
                    json.dump(self.args_dict, json_file, indent=4, sort_keys=True, cls=CustomJSONserializer)
                self.args_dict = None

        if validation_loss < self.prev_validation_loss:
            self.prev_validation_loss = validation_loss
            unwrapped_model = accelerator.unwrap_model(model)
            # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
            # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
            # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
            # For Zero Stages 1 and 2, models are saved as usual in the output directory.
            # The model name saved is `pytorch_model.bin`

            unwrapped_model.save_pretrained(
                self.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                # max_shard_size="50MB", # set to a low value to debug with sharded GPT2. Normally keep it commented
                safe_serialization=False
            )
            print(f'Model saved in {self.output_dir}')

        accelerate.utils.wait_for_everyone()


def prepare_formal_natural_inputs(formal_texts, natural_texts, tokenizer, return_natural_inputs=True):
    """
    Prepare inputs for the model by tokenizing and converting to tensors on right device
    """
    # Encode formal to natural
    formal_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
    # Ensure inputs are on the right device
    formal_inputs = {k: v.to(PartialState().device) for k, v in formal_inputs.items()}

    # This has to be deterministic else some process will wait for the others!
    if return_natural_inputs:
        natural_inputs = tokenizer(natural_texts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        # Ensure inputs are on the right device
        natural_inputs = {k: v.to(PartialState().device) for k, v in natural_inputs.items()}
    else:
        natural_inputs = None
        
    return formal_inputs, natural_inputs
        

def create_val_metrics_string(metrics):
    return f'val_rec_l/rf: {metrics["recon_loss"]:.3f}/ {metrics["rf_recon_loss"]:.3f}, ' \
           f'val_enc_l/rf: {metrics["enc_loss"]:.3f}/ {metrics["rf_enc_loss"]:.3f}, ' \
           f'val_perp_l/rf: {metrics["log_perplexity_loss"]:.3f}/ {metrics["rf_log_perplexity_loss"]:.3f} ' \
           f'val_dec_l_nat/rf: {metrics["dec_los_on_nat"]:.3f}/ {metrics["rf_dec_los_on_nat"]:.3f}'


def run_validation(
    model: AutoEncoderLLM, accelerator: Accelerator, tokenizer, 
    non_rephrased_dataloader, rephrased_dataloader, 
    df_filename,
    model_kwargs: Optional[Dict] = None, max_num_batches=None,
    padding_type='',
    fl_init_end_toks=('', ''),
    nl_init_end_toks=('', '')
):
    non_rf_avg_encoder_loss, non_rf_avg_perplexity_loss, non_rf_avg_decoder_loss, non_rf_avg_dec_los_on_nat, non_rf_df = \
        compute_loss_and_df(model, accelerator, tokenizer, data_loader=non_rephrased_dataloader,
                            model_kwargs=model_kwargs, max_num_batches=max_num_batches, padding_type=padding_type,
                            fl_init_end_toks=fl_init_end_toks, nl_init_end_toks=nl_init_end_toks)
    rf_avg_encoder_loss, rf_avg_perplexity_loss, rf_avg_decoder_loss, rf_avg_dec_los_on_nat, rf_df = \
        compute_loss_and_df(model, accelerator, tokenizer, data_loader=rephrased_dataloader, model_kwargs=model_kwargs,
                            max_num_batches=max_num_batches, padding_type=padding_type,
                            fl_init_end_toks=fl_init_end_toks, nl_init_end_toks=nl_init_end_toks)
        
    metrics = {
        "recon_loss": non_rf_avg_decoder_loss, "enc_loss": non_rf_avg_encoder_loss,
        "log_perplexity_loss": non_rf_avg_perplexity_loss,
        "rf_recon_loss": rf_avg_decoder_loss, "rf_enc_loss": rf_avg_encoder_loss,
        "rf_log_perplexity_loss": rf_avg_perplexity_loss, 'dec_los_on_nat': non_rf_avg_dec_los_on_nat,
        'rf_dec_los_on_nat': rf_avg_dec_los_on_nat
    }

    if PartialState().is_main_process:    
        marker_row = pd.DataFrame(
            {column: "" if column != 'formal_target'
                else "---- START OF REPHRASED NL DATAFRAME ----" for column in non_rf_df.columns},
            index=[0])

        # Concatenate the first DataFrame, marker row, and the second DataFrame
        combined_df = pd.concat([non_rf_df, marker_row, rf_df], ignore_index=True)

        combined_df.to_csv(df_filename, index=False, encoding='utf-8')
        
    return metrics


def as_dict(**kwargs):
    return kwargs


# run on last batch without gather
def decode_logits_or_inputs(tokenizer, logits_or_inputs, compress):
    if logits_or_inputs is None:
        return None
    if isinstance(logits_or_inputs, dict):
        if "inputs_embeds" in logits_or_inputs:
            # cannot be decoded
            return None
        token_ids = logits_or_inputs["input_ids"]
    elif logits_or_inputs.ndim == 3:
        token_ids = torch.argmax(logits_or_inputs, dim=-1)
    else:
        # labels are not a dict, but a tensor of shape (batch_size, seq_len)
        token_ids = logits_or_inputs
    assert token_ids.ndim == 2
    token_ids[token_ids == -100] = tokenizer.wait_token_id
    text = tokenizer.batch_decode(token_ids.cpu(), skip_special_tokens=False)
    if compress:
        return batch_compress_text_forwaiting_and_eot_tokens(text, eot_token=tokenizer.eos_token)
    return text


def compute_loss_and_df(model: AutoEncoderLLM, accelerator: Accelerator, tokenizer, data_loader,
                        model_kwargs: Optional[Dict] = None, max_num_batches=None, padding_type='',
                        fl_init_end_toks=None, nl_init_end_toks=None):
    """
    Compute loss on given dataloader
    
    This does not handle the situation at the end of the dataloader well (due to duplicated data)
    
    Returns:
        
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    perplexity_losses = None
    encoder_losses = None
    decoder_losses = None
    decoder_losses_from_natural = None
    sizes = None
    
    def combine_handling_none(x, val):
        """combine array x and value, handling x=None"""
        if x is None:
            return np.array([val])
        return np.concatenate([x, [val]])
    
    model.eval()
    with torch.no_grad():
        is_first_batch = True
        for batch in itertools.islice(data_loader, max_num_batches):
            formal_texts, natural_texts = apply_start_end_tags(
                batch['formal'], batch['natural'], fl_init_end_toks, nl_init_end_toks)
            
            formal_inputs, natural_inputs = prepare_formal_natural_inputs(
                formal_texts, natural_texts, tokenizer=tokenizer,
            )

            model_outputs, model_inputs = model(formal_inputs=formal_inputs, natural_inputs=natural_inputs,
                                                **model_kwargs, return_inputs=is_first_batch, padding_type=padding_type,
                                                also_decode_natural=True)
            if model_outputs.log_perplexity_loss is not None:
                perplexity_loss = numpify(accelerator.gather(model_outputs.log_perplexity_loss))
                perplexity_losses = combine_handling_none(perplexity_losses, perplexity_loss)
            if model_outputs.encoder_outputs is not None:
                encoder_loss = numpify(accelerator.gather(model_outputs.encoder_outputs.loss))
                encoder_losses = combine_handling_none(encoder_losses, encoder_loss)
            if model_outputs.decoder_outputs is not None:
                decoder_loss = numpify(accelerator.gather(model_outputs.decoder_outputs.loss))
                decoder_losses = combine_handling_none(decoder_losses, decoder_loss)
            if model_outputs.decoder_outputs_from_natural is not None:
                decoder_loss_from_natural = numpify(accelerator.gather(model_outputs.decoder_outputs_from_natural.loss))
                decoder_losses_from_natural = combine_handling_none(decoder_losses_from_natural,
                                                                    decoder_loss_from_natural)
            # not working as of now
            # ones_for_batch = torch.ones(len(formal_texts), device=formal_inputs["input_ids"].device)
            # sizes = combine_handling_none(sizes, accelerator.gather_for_metrics(ones_for_batch).cpu().numpy())
            
            if is_first_batch:
                encoder_inputs_decoded = decode_logits_or_inputs(tokenizer, model_inputs.encoder_inputs,
                                                                 compress=True)
                encoder_targets_decoded = decode_logits_or_inputs(tokenizer, model_inputs.encoder_targets,
                                                                  compress=True)
                encoder_outputs_decoded = decode_logits_or_inputs(tokenizer, model_outputs.encoder_logits(),
                                                                  compress=True)
                
                decoder_inputs_decoded = decode_logits_or_inputs(tokenizer, model_inputs.decoder_inputs, compress=True)
                decoder_targets_decoded = decode_logits_or_inputs(tokenizer, model_inputs.decoder_targets,
                                                                  compress=True)
                decoder_outputs_decoded = decode_logits_or_inputs(tokenizer, model_outputs.decoder_logits(),
                                                                  compress=True)
                decoder_output_from_natural_decoded = decode_logits_or_inputs(
                    tokenizer, model_outputs.decoder_logits_from_natural(), compress=True)
                
            is_first_batch = False
    
    avg_perplexity_loss = np.average(perplexity_losses, weights=sizes) if perplexity_losses is not None else 0
    avg_encoder_loss = np.average(encoder_losses, weights=sizes) if encoder_losses is not None else 0
    avg_decoder_loss = np.average(decoder_losses, weights=sizes) if decoder_losses is not None else 0
    decoder_losses_from_natural = np.average(decoder_losses_from_natural, weights=sizes)\
        if decoder_losses_from_natural is not None else 0

    df = make_pandas_dataframe(**{
        'encoder_input': encoder_inputs_decoded,
        'encoder_target': encoder_targets_decoded,
        'encoder_output': encoder_outputs_decoded,
        
        'decoder_input': decoder_inputs_decoded,
        'decoder_target': decoder_targets_decoded,
        'decoder_output': decoder_outputs_decoded,
        'decoder_output_from_natural': decoder_output_from_natural_decoded,
        
        'avg_encoder_loss': avg_encoder_loss,
        'avg_perplexity_loss': avg_perplexity_loss,
        'avg_decoder_loss': avg_decoder_loss,
        'decoder_losses_from_natural': decoder_losses_from_natural
    })

    return avg_encoder_loss, avg_perplexity_loss, avg_decoder_loss, decoder_losses_from_natural, df
