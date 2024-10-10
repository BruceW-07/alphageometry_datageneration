import io
from pathlib import Path
import random
from ruamel.yaml import YAML


# Read YAML file and get a random problem
def get_random_problem(file_path, serializer):
    with open(file_path, 'r') as file:
        problems = serializer.load(file)
        chosen_prob = random.choice(problems)
        prob_text_natural = chosen_prob['problem']['description']
        prob_text_formal_stream = io.StringIO()
        serializer.dump(chosen_prob, prob_text_formal_stream)

    return f'{prob_text_natural}\n```\n{prob_text_formal_stream.getvalue()}```'


class PromptGenerator:
    def __init__(self, prob_intro_tmplt_path, example_probs_path):
        self.prob_intro_tmplt_path, self.example_probs_path, = prob_intro_tmplt_path, example_probs_path
        self.serializer = YAML()
        self.serializer.preserve_quotes = True

    def prepare_intro_prompt(self, nl_problem, verbalizer):
        prob_intro_tmplt_path = Path(self.prob_intro_tmplt_path)
        prob_intro_tmplt = prob_intro_tmplt_path.read_text()

        example_problem = get_random_problem(self.example_probs_path, self.serializer)
        pre_def_const = verbalizer.get_comment_like()

        prob_intro_prompt = prob_intro_tmplt.format(example_problem=example_problem,
                                                    predefined_constraints=pre_def_const,
                                                    nl_prob=nl_problem)

        return prob_intro_prompt

    def prepare_error_prompt(self, nl_prob, fl_prob, error_messages):
        return (f'You formalized the natural language problem as follows.\n{nl_prob}\n'
                f'```\n{fl_prob}\n```\n'
                f'However you used some symbols incorrectly. The detailed error messages are as '
                f'follows\n{error_messages}')

    def prepare_match_prompt(self, nl_problem, rephrased_nl_problem):
        prompt = (f'Here are two description of two geometric concepts. Think carefully and asses if they are the '
                  f'same. If they are same return "Yes" else "No", nothing else. Your response will be machine '
                  f'interpreted therefore returning unformatted "Yes" or "No" is essential.'
                  f'\nDescription 1:\n{nl_problem}\nDescription 2:\n{rephrased_nl_problem}')
        return prompt
