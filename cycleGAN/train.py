import os
import math
import re
from torch.utils.data import DataLoader

from accelerate import Accelerator
from hf_dataset import (AlwaysSameElementDataset, CombinedDataset, MixedDatasetSampler, prepare_data,
                        shuffle_sentences_in_batch)
from model_preparation import load_model
from transformers import get_scheduler
from torch.optim import AdamW
from my_utils.generic_utils import get_process_cuda_memory_info, print_proc0, print_model_device_distribution, \
    ProgressBar, get_project_out_dir, apply_start_end_tags, get_cmd_args

from my_utils.training_utils import as_dict, create_val_metrics_string, Checkpointer, prepare_formal_natural_inputs,\
    run_validation
from utils import freeze_params

import accelerate

import wandb
os.environ.setdefault("WANDB_PROJECT", "alphageom_new")
wandb.require("core")

from utils import get_username
from my_utils.hf_wrapper import debug_on_error

@debug_on_error
def main(args):
    # Initialize Accelerator
    accelerator = Accelerator(log_with="all")
    if get_username() == "mmordig":
        accelerator.init_trackers("alphageom_autoencoder", config=vars(args))
        # pass
    valid_recon_save_path, chkpt_dir = get_project_out_dir(args, accelerator.is_main_process)
    print(f"Outputting to dirs {valid_recon_save_path} and {chkpt_dir}")
    checkpointer = Checkpointer(chkpt_dir, args)

    seed = 42
    accelerate.utils.set_seed(seed)

    # Load tokenizer and models
    ae_model, tokenizer, wait_id = load_model(args.model_name, wait_token=args.wait_tok,
                                              use_pretrained=args.is_pretrained,
                                              use_perplexity_loss=args.use_perplexity_loss,
                                              use_decoder=args.use_decoder, use_encoder=args.use_encoder,
                                              fl_init_end_toks=[args.formal_init_tok, args.formal_end_tok],
                                              nl_init_end_toks=[args.natural_init_tok, args.natural_end_tok])

    wrap_dataset = AlwaysSameElementDataset if args.overfitting else lambda x: x
    # Prepare dataset and dataloader
    # synthetic_nl_fl_file = 'nl_fl.csv'
    # synthetic_nl_fl_file = 'nl_fl_long.csv'
    synthetic_nl_fl_file = 'svg_fl_long.csv'
    non_rephrased_dataset = prepare_data(
        os.path.join(args.dataset_dir, synthetic_nl_fl_file) , seed=seed, nrows=args.nrows_nonrephrased)
    # rephrased_file_name = 'rephrased-nl_fl_dataset_all.jsonl'
    # rephrased_file_name = 'all_rephrased_chunks.csv'
    rephrased_file_name = 'rephrases-10k.csv'
    rephrased_dataset = prepare_data(
        os.path.join(args.dataset_dir, rephrased_file_name), seed=seed, nrows=args.nrows_rephrased,
        colnames={"formal": "fl_statement", "natural": "rephrase", "total_token_lens": "total_token_lens"})
    rephrased_dataset = rephrased_dataset.filter(lambda x: x["total_token_lens"] < 1500 and x["natural"])
    # import ipdb; ipdb.set_trace()
    rephrased_dataset = rephrased_dataset.map(
        lambda x: {"natural": [re.sub(r'\r\n|\r|\n', '', text) for text in x["natural"]]},
        batched=True,
    )
    rephrased_dataset = rephrased_dataset.select_columns(["formal", "natural"])

    if args.rephrased_ratio > 0:
        print(f"Using {args.rephrased_ratio} rephrased data")
        train_datasets = [non_rephrased_dataset["train"], rephrased_dataset["train"]]
        assert set(train_datasets[0].column_names) == set(train_datasets[1].column_names)
        train_dataset = CombinedDataset(*train_datasets)
        train_sampler = MixedDatasetSampler(
            [len(x) for x in train_datasets], args.rephrased_ratio
        )
        del train_datasets
    else:
        train_dataset = non_rephrased_dataset["train"]
        train_sampler = None
    train_dataset = wrap_dataset(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True,
                                  shuffle=False)

    val_dataset = wrap_dataset(non_rephrased_dataset["validation"])
    val_rephrased_dat = wrap_dataset(rephrased_dataset["validation"])

    valid_sampler = None # not needed, handled by accelerate
    val_rephrased_samp = None
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True,
                                shuffle=False)
    val_rephrased_dataloader = DataLoader(val_rephrased_dat, batch_size=args.batch_size, sampler=val_rephrased_samp)

    # Optimizers
    if args.enc_is_trainable and args.dec_is_trainable:
        optimizer = AdamW(ae_model.parameters(), lr=2e-5 * 4)
    elif args.enc_is_trainable:
        optimizer = AdamW(ae_model.encoder.parameters(), lr=2e-5 * 4)
    elif args.dec_is_trainable:
        optimizer = AdamW(ae_model.decoder.parameters(), lr=2e-5 * 4)
    
    # Learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        # name="linear",
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # ae_model.load_weights(args.enc_resume_path, args.dec_resume_path)

    # import ipdb; ipdb.set_trace()
    ae_model, optimizer, lr_scheduler, train_dataloader, val_dataloader, val_rephrased_dataloader = accelerator.prepare(
        ae_model, optimizer, lr_scheduler, train_dataloader, val_dataloader, val_rephrased_dataloader
    )
    # All processes should do the following! don't wrap the if main process condition above!
    # import ipdb; ipdb.set_trace()
    ae_model.load_weights(args.enc_resume_path, args.dec_resume_path)
    ae_model.freeze_perplexity_model()
    if not args.enc_is_trainable:
        freeze_params(ae_model.encoder)
    if not args.dec_is_trainable:
        freeze_params(ae_model.decoder)

    # Training loop
    ae_model.train()
    val_update_str = f'val_update: N/A'
    for epoch in range(args.num_epochs):
        pbar = ProgressBar(train_dataloader, accelerator)
        for batch_idx, batch in enumerate(pbar):
            # A list of variable length of string. So is natural texts
            formal_texts, natural_texts = apply_start_end_tags(
                batch['formal'], batch['natural'], [args.formal_init_tok, args.formal_end_tok],
                [args.natural_init_tok, args.natural_end_tok])
            natural_texts = shuffle_sentences_in_batch(natural_texts) if args.shuffle_sentences else natural_texts

            formal_inputs, natural_inputs = prepare_formal_natural_inputs(
                formal_texts, natural_texts, tokenizer=tokenizer,
                return_natural_inputs=batch_idx % math.ceil(1/(args.grounding_prob + 1e-7)) == 0
            )

            ae_model.train()
            model_outputs, _ = ae_model(formal_inputs=formal_inputs, natural_inputs=natural_inputs,
                                        wait_token_id=wait_id, pad_token_id=tokenizer.pad_token_id,
                                        padding_type=args.padding_type, also_decode_natural=True)
            total_loss = model_outputs.total_loss(enc_loss_weight=args.enc_loss_weight)

            accelerator.backward(total_loss)
            # import ipdb; ipdb.set_trace()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_metrics = {
                "lg_perp_l": model_outputs.log_perplexity_loss or 0,
                "recon_l": model_outputs.decoder_loss(),
                "enc_l": model_outputs.encoder_loss(),
                "dec_nat_l": model_outputs.decoder_loss_on_nat_lang_input()
            }

            if batch_idx % args.validate_every == args.validate_every - 1:
                
                df_filename = os.path.join(valid_recon_save_path, f'{epoch}_{batch_idx}_fl_fl.csv')
                val_metrics = run_validation(
                    model=ae_model, accelerator=accelerator, tokenizer=tokenizer, 
                    non_rephrased_dataloader=val_dataloader, rephrased_dataloader=val_rephrased_dataloader,
                    df_filename=df_filename,
                    model_kwargs=as_dict(wait_token_id=wait_id, pad_token_id=tokenizer.pad_token_id),
                    max_num_batches=args.valid_for_batches, padding_type=args.padding_type,
                    fl_init_end_toks=[args.formal_init_tok, args.formal_end_tok],
                    nl_init_end_toks=[args.natural_init_tok, args.natural_end_tok])
                val_update_str = "val_update: " + create_val_metrics_string(val_metrics)
                accelerator.log(
                    {f"val/{k}": v for k, v in val_metrics.items()}
                    |
                    {f"train/{k}": v for k, v in train_metrics.items()}
                    |
                    {f"epoch": epoch + batch_idx / len(train_dataloader),
                     "step": epoch * len(train_dataloader) + batch_idx}
                )
                
                ae_model.train()
                if (batch_idx + 1) % args.chkpt_bst_mdl_every == 0:
                    loss_for_saving = val_metrics['rf_dec_los_on_nat']
                    if loss_for_saving == 0:
                        loss_for_saving = val_metrics["enc_loss"] + val_metrics["recon_loss"]

                    checkpointer.checkpoint(accelerator, ae_model, loss_for_saving)
                
            train_update_str = " ".join([f"{k}:{v:.3f}" for k, v in train_metrics.items()])
            pbar.set_description(f'Epoch {epoch+1}/{args.num_epochs}: {val_update_str} {train_update_str}')

    accelerator.end_training()
    print(f"Training completed. A total of {epoch+1} epoch(s)")


if __name__ == "__main__":
    args = get_cmd_args()
    main(args)
