import torch
import os
import tqdm
import re
import pandas as pd
import json
from pathlib import Path
import accelerate
import sys
from utils import get_comma_separated_strings, get_hostname, get_username


class ProgressBar:
    """Progress bar on main process only, otherwise normal iterable"""
    def __init__(self, iterable_obj, accelerator):
        self.accelerator = accelerator
        self.local_rank = accelerator.state.process_index
        if self.local_rank == 0:
            self.pbar = tqdm.tqdm(iterable_obj)
        else:
            self.pbar = iterable_obj

    def __iter__(self):
        return self.pbar.__iter__()

    def __next__(self):
        return self.pbar.__next__()

    def update(self, n=1):
        if self.local_rank == 0:
            self.pbar.update(n)

    def set_description(self, desc):
        if self.local_rank == 0:
            self.pbar.set_description(desc)


class CustomJSONserializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            # Convert Path object to its absolute path string representation
            return str(obj.absolute())
        else:
            try:
                return super().default(obj)
            except TypeError:
                # Fallback to string representation for unserializable objects
                print(f"WARNING: Object of type {type(obj).__name__} is not serializable. "
                      f"Replacing with its string representation.")

                return str(obj)


def apply_start_end_tags(fl_text_list, nl_text_list, fl_init_end_toks, nl_init_end_toks):
    assert len(fl_text_list) == len(nl_text_list)
    for i, fl_txt in enumerate(fl_text_list):
        fl_text_list[i] = f'{fl_init_end_toks[0]}{fl_txt}{fl_init_end_toks[1]}'
        nl_text_list[i] = f'{nl_init_end_toks[0]}{nl_text_list[i]}{nl_init_end_toks[1]}'

    return fl_text_list, nl_text_list


def get_project_out_dir(args, is_main_process):
    chkpt_paths = ['', '']
    if is_main_process:
        mdl_dir = args.model_name
        if args.use_decoder and not args.use_encoder:
            mdl_dir += '_dec_only'
        elif args.use_encoder and not args.use_decoder:
            mdl_dir += '_enc_only'
        else:
            mdl_dir += '_full_ae'
        if args.use_perplexity_loss:
            mdl_dir += '+perplexity'

        if not args.enc_is_trainable:
            mdl_dir += '_frozen_enc'

        if not args.dec_is_trainable:
            mdl_dir += '_frozen_dec'
        mdl_dir_with_count = mdl_dir

        for count in range(999):
            mdl_output_home = os.path.join(args.output_path, mdl_dir_with_count)
            valid_recon_save_path = os.path.join(mdl_output_home, 'validation_outputs')
            chkpt_dir = os.path.join(mdl_output_home, 'checkpoints')
            if os.path.exists(mdl_output_home):
                mdl_dir_with_count = mdl_dir + f'_{count}'
            else:
                break

        os.makedirs(valid_recon_save_path, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_paths = [str(valid_recon_save_path), str(chkpt_dir)]

    valid_recon_save_path, chkpt_dir = accelerate.utils.broadcast_object_list(chkpt_paths, from_process=0)

    args.valid_recon_save_path = valid_recon_save_path
    args.chkpt_dir = chkpt_dir

    return valid_recon_save_path, chkpt_dir


def compress_text_forwaiting_and_eot_tokens(input_text, wait_token='<w>', eot_token='<|endoftext|>'):
    """
    Compresses texts by counting leading <w> tokens and trailing <|endoftext|> tokens,
    and then formats the string accordingly.

    Example:
        input: '<w> <w> A B C = Triangle A B C <|endoftext|><|endoftext|><|endoftext|>'
        output: '2<w> A B C = Triangle A B C 3<|endoftext|>'
    """
    # Adjusted regex to properly match the required pattern
    pattern = r'^((' + re.escape(wait_token) + r')\s*)+'
    match = re.match(pattern, input_text)
    if match:
        # Count occurrences of each token type
        wait_section = match.group() or ''
        count_w = wait_section.count(wait_token)
        rest_of_text = input_text[match.span()[1]:]

        # Count trailing eot tokens accurately
        count_eot = 0
        while rest_of_text.endswith(eot_token):
            count_eot += 1
            rest_of_text = rest_of_text[:-len(eot_token)]

        # Format output string with counts
        result_text = f"{count_w}{wait_token} {rest_of_text.strip()} {count_eot}{eot_token}"
        return result_text.strip()

    return input_text  # Return original text if no matches


def batch_compress_text_forwaiting_and_eot_tokens(input_list_text, wait_token='<w>', eot_token='<|endoftext|>'):
    if input_list_text is None:
        return None
    return [compress_text_forwaiting_and_eot_tokens(re.sub(r'\r\n|\r|\n', '<new_line>', text),
                                                    wait_token, eot_token) for text in input_list_text]


def get_process_cuda_memory_info():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))  # Defaults to 0 if LOCAL_RANK is not set
    memory_info = {}

    if torch.cuda.is_available():
        device_info = {
            'device_name': torch.cuda.get_device_name(local_rank),
            'memory_allocated_gb': torch.cuda.memory_allocated(local_rank) / 1e9,  # Convert bytes to GB
            'memory_reserved_gb': torch.cuda.memory_reserved(local_rank) / 1e9,  # Convert bytes to GB
            'total_memory_gb': torch.cuda.get_device_properties(local_rank).total_memory / 1e9,  # Convert bytes to GB
        }

        # Calculate used and free memory
        free_memory = torch.cuda.memory_reserved(local_rank) / 1e9
        total_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        used_memory = total_memory - free_memory

        device_info['used_memory_gb'] = used_memory
        device_info['free_memory_gb'] = free_memory

        # Peak memory usage
        mem_stats = torch.cuda.memory_stats(local_rank)
        device_info['peak_memory_allocated_gb'] = mem_stats['allocated_bytes.all.peak'] / 1e9
        device_info['peak_memory_reserved_gb'] = mem_stats['reserved_bytes.all.peak'] / 1e9

        memory_info[f"cuda:{local_rank}"] = device_info
    else:
        memory_info['error'] = "No CUDA devices available."

    return memory_info


def numpify(t):
    # handles bfloat16 issue
    # see transformers.trainer_pt_utils.nested_numpify
    t = t.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()


def print_proc0(msg):
    """print only on process 0"""
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(msg)


def make_pandas_dataframe(**kwargs):
    data_dict = {}
    max_length = 0

    # First pass to replace None and determine maximum list length
    for key, value in kwargs.items():
        if value is None:
            data_dict[key] = "_"
        else:
            # Replace list Nones and find max length
            if isinstance(value, list):
                # Replace None in lists with '_'
                cleaned_list = ['_' if v is None else v for v in value]
                data_dict[key] = cleaned_list
                max_length = max(max_length, len(cleaned_list))
            else:
                data_dict[key] = value

    # Second pass to adjust lists to the maximum length
    for key, value in data_dict.items():
        if isinstance(value, list) and len(value) < max_length:
            # Extend lists shorter than the max length with '_'
            data_dict[key] = value + ['_'] * (max_length - len(value))

    return pd.DataFrame(data_dict)


def print_model_device_distribution(accelerator, model, model_name):
    device_params_count = {}  # Dictionary to hold device counts

    # Iterate through all parameters and count them per device
    for parameter in model.parameters():
        device = parameter.device
        if device in device_params_count:
            device_params_count[device] += parameter.numel()  # Add number of elements in parameter
        else:
            device_params_count[device] = parameter.numel()

    # Print the process rank and distribution of parameters across devices
    if accelerator.distributed_type.lower() == 'deepspeed':
        print('WARNING: Parameter distribution will not be visible like this!')
    print(f"Rank: {accelerator.state.process_index}, {model_name} is distributed across the following devices:")
    for device, count in device_params_count.items():
        print(f"Device: {device}, Number of Parameters: {count:,}")


def get_cmd_args():
    if get_hostname() == "mikado":
        os.environ["ALPHA_GEOM_DATASET"] = "/home/mmordig/reinforcement/alphageometry/cycleGAN/runs/datasets/arithmetic"
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--wait_tok', type=str, default='<w>')
    parser.add_argument('--formal_init_tok', type=str, default='<fl>')
    parser.add_argument('--formal_end_tok', type=str, default='</fl>')
    parser.add_argument('--natural_init_tok', type=str, default='<nl>')
    parser.add_argument('--natural_end_tok', type=str, default='</nl>')
    parser.add_argument('--validate_every', type=int, default=100, help='Validate every these many training steps '
                                                                        '(reset per epoch)')
    parser.add_argument('--valid_for_batches', type=int, default=10, help='Validate for these many batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU! when deepspeed is enabled '
                                                                   '(for model pipelining), it is divided by the number'
                                                                   ' of gpus for microbatching')
    parser.add_argument('--dataset_dir', type=Path, default=None, help='Path to dataset directory')
    parser.add_argument('--chkpt_bst_mdl_every', type=int, default=10,
                        help='Checkpoint model every these many validation (!) steps if validation result improved. '
                             'Negative value skips this')
    parser.add_argument('--output_path', type=str, default=None, help='path to save training stats and models')
    parser.add_argument('--enc_resume_path', type=str, default=None, help='path to load encoder weights from. '
                                                                          'e.g., a previous run')
    parser.add_argument('--dec_resume_path', type=str, default=None, help='path to load decoder weights from. '
                                                                          'e.g., a previous run')
    parser.add_argument('--grounding_prob', type=float, default=0.5, help='introduce encoder NL labels every ceil(1/x)'
                                                                          ' batches')
    parser.add_argument('--enc_loss_weight', type=float, default=2, help='scale encoder loss by this factor')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name to load, e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',"
                             "'meta-llama/Meta-Llama-3.1-8B', "
                             "'meta-llama/Llama-2-7b-hf'")
    parser.add_argument('--overfitting', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False,
                        help="whether to overfit on a single batch (same across all GPUs)")
    parser.add_argument('--is_pretrained', type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)
    parser.add_argument('--use_decoder', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--use_encoder', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--enc_is_trainable', type=lambda x: True if x.lower() in ['true', '1'] else False,
                        default=True)
    parser.add_argument('--dec_is_trainable', type=lambda x: True if x.lower() in ['true', '1'] else False,
                        default=True)
    parser.add_argument('--shuffle_sentences',
                        type=lambda x: True if x.lower() in ['true', '1'] else False,
                        default=False)
    parser.add_argument('--use_perplexity_loss',
                        type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)
    parser.add_argument('--nrows_nonrephrased', type=int, default=None,
                        help='Number of rows to load from the non-rephrased dataset before splitting into '
                             'train/val/test, defaults to all')
    parser.add_argument('--nrows_rephrased', type=int, default=None,
                        help='Number of rows to load from the rephrased dataset before splitting into '
                             'train/val/test, defaults to all')
    parser.add_argument('--rephrased_ratio', type=float, default=0, help='Ratio of picking from rephrased dataset')
    parser.add_argument('--padding_type', type=str,
                        default='copy_target',
                        help='Padding mode to use. Either "copy_target" or "pad_tok" as possible')
    print(f"Unparsed arguments: {get_comma_separated_strings(sys.argv)}")
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = os.environ.get("ALPHA_GEOM_OUTPUT_PATH",
                                          '/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/')
        if get_username() == "mmordig":
            args.output_path = "runs/"
        print(f"Output path not provided, using {args.output_path}")
    if args.dataset_dir is None:
        args.dataset_dir = Path(os.environ.get("ALPHA_GEOM_DATASET",
                                               '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/'))
    assert 0 <= args.rephrased_ratio, f"got invalid {args.rephrased_ratio=}"
    args.chkpt_bst_mdl_every *= args.validate_every
    if args.use_decoder and not args.use_encoder:
        assert not args.use_perplexity_loss
        assert args.grounding_prob >= 1  # you need the grounding always as these are the inputs
    if args.use_encoder and not args.use_decoder:
        assert args.grounding_prob >= 1, f'We need the natural language targets when using encoder only model.'

    if args.enc_resume_path is not None and args.dec_resume_path is not None:
        # The tokenizer is pretrained anyway!
        args.is_pretrained = False  # no point loading language model pretraining as we will load checkpoint anyway
    print_proc0(f"Got arguments: {args}")
    return args


if __name__ == '__main__':
    # Example usage
    # gpu_memory_info = get_process_cuda_memory_info()
    # print(gpu_memory_info)

    # print(compress_text_forwaiting_and_eot_tokens(
    #     '<w> <w><w> A B C = Triangle A B C <|endoftext|><|endoftext|><|endoftext|>'))
    # print(compress_text_forwaiting_and_eot_tokens('<w><w> A B C = Triangle A B C </s></s>', eot_token='</s>'))
    text = '<w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w>' \
           '<w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w><w>' \
           '<w><w><w><w><|begin_of_text|><fl>A B C D E = pentagon A B C D E; F = lc_tangent F E C; G H I = ' \
           'r_triangle G H I</fl><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>' \
           '<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>'
    print(compress_text_forwaiting_and_eot_tokens(text, eot_token='<|end_of_text|>'))