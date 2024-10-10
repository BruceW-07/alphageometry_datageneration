import omnifig as fig
from openai import AzureOpenAI
import os
from verbalize.verbalize_constraints import PreDefConstVerbalizer
from tool_generation.prompt_gen.chat_manager import ChatHistManager
from tool_generation.environment.draw_search_tree import DrawSearchTree
from prompt_gen.prompt_generator import PromptGenerator
from tool_generation.environment.auto_formalization_env import formalize_s0


@fig.script('formalize_geo_probs')
def formalize_geo_probs(cfg: fig.Configuration):
    nl_prob = 'Two circles G1 and G2 intersect at two points M and N.'  # Perhaps should be read from a file
    show_prompt = cfg.pull('show_prompt', default=True)
    dry_run = cfg.pull('dry-run', False)
    if dry_run:
        print(f'Dry run: not actually saving anything.')
        show_prompt = True

    use_pbar = cfg.pull('pbar', True, silent=True)
    print_freq = 0 if use_pbar else cfg.pull('print-freq', 10)

    api_key = cfg.pull('api-key', os.environ.get('OPENAI_API_KEY', None), silent=True)
    if api_key is None:
        raise ValueError(f'No OPENAI API key found, pass argument using "--api-key" or set env var "OPENAI_API_KEY"')

    max_tokens = cfg.pull('max_tokens', 2000)

    api_version = cfg.pull('api-version', None, silent=True)
    azure_endpoint = cfg.pull('api-base', None, silent=True)
    model_name = cfg.pull('model-name', 'gpt-3.5-turbo', silent=True)

    overwrite = cfg.pull('overwrite', False)

    tree_drawer = DrawSearchTree()
    pre_def_const_file = 'prompt_gen/atomic_constraints.yml'
    verbalizer = PreDefConstVerbalizer(pre_def_const_file)
    prompt_gen = PromptGenerator('prompt_gen/prob_intro_template.txt',
                                 'prompt_gen/example_problem.yml')
    call_llm_manager = ChatHistManager(api_key, api_version, azure_endpoint, model_name, max_tokens, is_dummy=dry_run)
    fl_prob = formalize_s0(0,
                           'C is a point on the perpendicular bisector of AB. Prove that AC = BC.',
                           cal_llm_manager=call_llm_manager, verbalizer=verbalizer, prompt_gen=prompt_gen,
                           parent_func='root', plotter=tree_drawer)
    print(fl_prob)


if __name__ == '__main__':
    fig.entry('formalize_geo_probs')