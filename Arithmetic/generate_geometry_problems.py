import sys
from email.policy import default

sys.path.append('../')
import os
import json
from pathlib import Path
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import random
import omnifig as fig

def modify_path_with_seed(original_path, seed_val):
    """
    Modifies the filename of an original Path by appending a seed value to the filename,
    while retaining the original directory.

    Parameters:
    original_path (Path): The original Path object.
    seed_val (int or str): The seed value to append to the filename.

    Returns:
    Path: A new Path object with the modified filename in the same directory.
    """
    # Extract the directory of the original path
    original_dir = original_path.parent

    # Extract the file name without the extension
    file_name = original_path.stem

    # Construct the new file name with the seed value
    new_file_name = f"{file_name}_seed_{seed_val}{original_path.suffix}"

    # Create the new path in the same directory
    new_path = original_dir / new_file_name

    return new_path

def random_lines_as_string(filename, num_lines=5):
    """
    Returns a specified number of random lines from a text file, formatted as a single string
    with lines separated by a single newline character.

    Parameters:
    filename (str): The path to the text file.
    num_lines (int): The number of random lines to return.

    Returns:
    str: A string containing the randomly selected lines, each separated by a newline.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Ensure the file has at least as many lines as requested
    if len(lines) < num_lines:
        raise ValueError("The file contains fewer lines than the requested number of random lines.")

    # Select num_lines random lines from the list and join them into a single string
    random_lines = random.sample(lines, num_lines)
    return ''.join(random_lines)  # Concatenate the list into a single string with newlines


@fig.script('gen_geo_probs')
def gen_geo_probs(cfg: fig.Configuration):
    show_prompt = cfg.pull('show_prompt', default=False)
    dry_run = cfg.pull('dry-run', False)
    if dry_run:
        print(f'Dry run: not actually saving anything.')
        show_prompt = True

    use_pbar = cfg.pull('pbar', True, silent=True)
    print_freq = 0 if use_pbar else cfg.pull('print-freq', 10)

    api_key = cfg.pull('api-key', os.environ.get('OPENAI_API_KEY', None), silent=True)
    if api_key is None:
        raise ValueError(f'No OPENAI API key found, pass argument using "--api-key" or set env var "OPENAI_API_KEY"')

    # client = OpenAI(
    #     base_url=cfg.pull('api-base', None, silent=True),
    #     api_key=api_key,
    # )
    client = AzureOpenAI(
        api_key=api_key,
        api_version=cfg.pull('api-version', None, silent=True),
        azure_endpoint=cfg.pull('api-base', None, silent=True),
    )
    model_name = cfg.pull('model-name', 'gpt-3.5-turbo', silent=True)

    overwrite = cfg.pull('overwrite', False)

    template_path = cfg.pull('template-path', default='Arithmetic/geo_prob_gen_template.txt')
    template_path = Path(template_path)
    template = template_path.read_text()

    # pbar = cfg.pull('pbar', not cfg.pull('no-pbar', False, silent=True), silent=True)

    max_tokens = cfg.pull('max-tokens', 2000)

    examples_root = Path(cfg.pull('examples_path', default='rephrase/o1_generated_nl_geo_problems.txt'))

    gen_attempts = cfg.pulls('n', 'gen_attempts', default=1)

    skip_confirm = cfg.pull('skip-confirm', False)
    while not skip_confirm:
        inp = input('Begin? ([y]/n) ').lower()
        if inp.startswith('n'):
            print('Ending without doing anything.')
        elif not len(inp) or inp.startswith('y'):
            break
        else:
            print('Try again.')

    seed = cfg.pull('seed', default=0)
    # path_itr = tqdm(paths) if pbar and len(paths) > 1 else paths
    n = 0
    outpath = Path(cfg.pull('outpath', default='new_problems.txt'))
    outpath = modify_path_with_seed(outpath, seed)
    if not overwrite and os.path.exists(outpath):
        print('Specified file with this seed already exist. Exiting ...')
        exit(0)
    writer = outpath.open('a')

    # item_itr = tqdm(df.iterrows(), total=len(df)) if pbar and len(paths) == 1 else df.iterrows()

    pbar = tqdm(total=gen_attempts) if use_pbar else None
    num_problems = cfg.pull('num_problems', default=20)
    print(f'creating {num_problems} problems per attempt')
    for i in range(gen_attempts):
        examples = random_lines_as_string(examples_root, num_lines=10)
        prompt = template.format(problems=examples, num_problems=num_problems)
        if show_prompt:
            print(f'Prompt:\n\n{prompt}\n\n')
            # show_prompt = False

        if dry_run:
            # print(f'Prompt: {prompt}')
            new_geo_probs = 'Dry run: no rephrased text'
        else:
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                # model="gpt-3.5-turbo",
                model=model_name,
                max_tokens=max_tokens,
            )
            new_geo_probs = response.choices[0].message.content

        if not new_geo_probs:
            if response.choices[0].finish_reason == 'content_filter':
                print(
                    f'Skipped due to content filter: {response.choices[0].content_filter_results}')  # and {response.choices[0].prompt_filter_results}')
                continue
            print(response)
            raise ValueError('No response from the API')


        writer.write(new_geo_probs + ('\n' if not new_geo_probs.endswith('\n') else ''))
        writer.flush()
        n += 1

        if print_freq and n % print_freq == 0:
            print(f'Processed {n} items')
        if pbar:
            pbar.update(1)


    writer.close()
    print(f'Finished writing to {outpath}')



if __name__ == '__main__':
    fig.entry('gen_geo_probs')

