import sys
sys.path.append('../')
import os
import json
from pathlib import Path
from openai import OpenAI, AzureOpenAI
from tqdm import tqdm
import pandas as pd
import hashlib
import omnifig as fig

from Arithmetic.verb import repo_root

def hash_item(item: dict[str, str]) -> str:
    if 'md5_item_code' in item:
        return item['md5_item_code']
    raw = json.dumps(item, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def load_existing_codes(path: Path) -> set[str]:
    if path.exists():
        existing = {json.loads(line).get('md5_item_code') for line in path.open('r')}
        return {code for code in existing if code is not None}
    return set()

@fig.script('count-rephrases')
def count_rephrases(cfg: fig.Configuration):
    path = cfg.pull('path')
    path = Path(path)

    completed_paths = list(path.glob('rephrased-*.jsonl'))
    available_paths = list(path.glob('*.csv')) + list(path.glob('*.CSV'))

    skip_available = cfg.pull('skip-available', False)

    completed = 0
    available = 0

    if not skip_available:
        for path in tqdm(available_paths):
            available += len(pd.read_csv(path))
    for path in tqdm(completed_paths):
        completed += sum(1 for _ in path.open('r'))
        
    print(f'Completed: {completed} items')
    if not skip_available:
        print(f'Available: {available} items')


@fig.script('rephrase')
def rephrase(cfg: fig.Configuration):

    show_prompt = False
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
        api_key = api_key,
        api_version = cfg.pull('api-version', None, silent=True),
        azure_endpoint = cfg.pull('api-base', None, silent=True),
    )
    model_name = cfg.pull('model-name', 'gpt-3.5-turbo', silent=True)

    overwrite = cfg.pull('overwrite', False)

    template_path = cfg.pull('template-path') # str(repo_root() / 'Arithmetic' / 'rephrase_template.txt')
    template_path = Path(template_path)
    template = template_path.read_text()

    # pbar = cfg.pull('pbar', not cfg.pull('no-pbar', False, silent=True), silent=True)

    max_tokens = cfg.pull('max-tokens', 2000)

    root = cfg.pull('path')
    root = Path(root)
    if root.is_dir():
        paths = list(root.glob('*.csv')) + list(root.glob('*.CSV'))
    else:
        paths = [root]
        root = None

    outpattern = cfg.pull('outpattern', '{path.parent}/rephrased-{path.stem}.jsonl', silent=True)

    max_num = cfg.pulls('n', 'max-num', default=None)

    print(f'Will save rephrases of {len(paths)} file/s')
    if max_num is not None:
        print(f'Will process at most {max_num} items.')

    skip_confirm = cfg.pull('skip-confirm', False)
    while not skip_confirm:
        inp = input('Begin? ([y]/n) ').lower()
        if inp.startswith('n'):
            print('Ending without doing anything.')
        elif not len(inp) or inp.startswith('y'):
            break
        else:
            print('Try again.')

    # path_itr = tqdm(paths) if pbar and len(paths) > 1 else paths
    n = 0
    path_itr = paths
    for path in path_itr:
        path = path.expanduser().absolute()
        if len(paths) == 1:
            print(f'Loading {path}')
        df = pd.read_csv(path)

        outpath = Path(outpattern.format(path=path))
        assert outpath.suffix == '.jsonl', 'Output path must be a jsonl file: {outpath}'
        outpath.parent.mkdir(parents=True, exist_ok=True)

        existing_codes = load_existing_codes(outpath) if not overwrite else set()

        writer = outpath.open('a')

        # item_itr = tqdm(df.iterrows(), total=len(df)) if pbar and len(paths) == 1 else df.iterrows()
        
        pbar = tqdm(total=len(df) if max_num is None else min(max_num, len(df))) if use_pbar else None
        for i, item in df.iterrows():
            item = item.to_dict()
            assert overwrite or 'rephrase' not in item, 'Item has already been rephrased'

            code = hash_item(item)

            if code in existing_codes:
                continue

            prompt = template.format(**item)
            if show_prompt:
                print(f'Prompt:\n\n{prompt}\n\n')
                show_prompt = False

            if dry_run:
                # print(f'Prompt: {prompt}')
                rephrased = 'Dry run: no rephrased text'
            else:
                response = client.chat.completions.create(
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}],
                    # model="gpt-3.5-turbo",
                    model=model_name,
                    max_tokens=max_tokens,
                )
                rephrased = response.choices[0].message.content

            if not rephrased:
                if response.choices[0].finish_reason == 'content_filter':
                    print(f'Skipped due to content filter: {response.choices[0].content_filter_results}')#and {response.choices[0].prompt_filter_results}')
                    continue
                print(response)
                raise ValueError('No response from the API')

            item['rephrase'] = rephrased.replace('\n', ' ')

            item['md5_item_code'] = code

            writer.write(json.dumps(item) + '\n')
            writer.flush()
            n += 1

            if print_freq and n % print_freq == 0:
                print(f'Processed {n} items')
            if pbar:
                pbar.update(1)

            if max_num <= n:
                print(f'Reached the max responses {max_num}')
                break

        writer.close()
        print(f'Finished writing to {outpath}')

        if max_num <= n:
            break


if __name__ == '__main__':
    fig.entry('rephrase')

