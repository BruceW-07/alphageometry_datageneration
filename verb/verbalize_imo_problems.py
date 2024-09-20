import sys
sys.path.append('..')
from pretty import pretty_nl
from prettier_print.prettier_proof_statements import translate_step
from verb.verbalize import IndependentStatementVerbalization
from cycleGAN import imo_problems
import pandas as pd


def format_geometric_description(description):
    # Split the input into main parts (before and after '?')
    parts = description.split('?')
    formatted_parts = []

    # Function to capitalize single letter point names
    def capitalize_points(clause):
        # Splitting each token in the clause and capitalizing if it is a single letter
        return ' '.join(word.upper() if len(word) == 1 and word.isalpha() else word for word in clause.split())

    # Process each part
    for part in parts:
        # Split into individual clauses
        clauses = part.split(';')
        # Capitalize point names in each clause and strip extra spaces
        capitalized_clauses = [capitalize_points(clause.strip()) for clause in clauses if clause.strip()]
        # Join the processed clauses back with semicolons
        formatted_parts.append('; '.join(capitalized_clauses))

    # Join the main parts with a '?' if there were originally two parts
    return ' ? '.join(formatted_parts) if len(formatted_parts) > 1 else formatted_parts[0]


# Function to read the file and extract the formal descriptions
def read_formal_descriptions(file_path):
    problem_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Pair problem names with their corresponding formal descriptions
        for i in range(0, len(lines), 2):
            problem_name = lines[i].strip()
            formal_description = lines[i + 1].strip()
            problem_dict[problem_name] = format_geometric_description(formal_description)

    return problem_dict


def verbalize_problem(verbalizer, fl_statement):
    fl_prob, goal_fl = fl_statement.split(' ? ')
    goal_fl = goal_fl.split(' ')
    nl_prob = verbalizer.problem_fl_2_nl(fl_prob.strip())

    capitalized_pt_names = [point_name.capitalize() for point_name in goal_fl[1:]]
    goal_fl[1:] = capitalized_pt_names
    pretty_goal = pretty_nl(goal_fl[0], goal_fl[1:])
    goal_nl = ' Prove that ' + translate_step(pretty_goal)
    return nl_prob + goal_nl


def get_definitions(csv_files):
    formal_natural_dict = {}

    for file in csv_files:
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            formal_term = row['formal'].strip()
            formal_term_wo_args = formal_term.split(' ')[0]
            natural_description = row['natural']

            # If the key is already present, retain the longest description
            if formal_term_wo_args in formal_natural_dict:
                if len(natural_description) > len(formal_natural_dict[formal_term_wo_args]):
                    formal_natural_dict[formal_term] = natural_description
            else:
                formal_natural_dict[formal_term_wo_args] = f'{formal_term} -> {natural_description}'

    return formal_natural_dict


def dict_to_str(dict_to_print, include_kay):
    description = ''
    for key, val in dict_to_print.items():
        if include_kay:
            description += f'{key}: {val}\n'
        else:
            description += f'{val}\n'

    if description.endswith('\n'):
        description = description[:-1]

    if not description.endswith('.'):
        description += '.'

    return description


def get_prompt(examples_dict, problem_statement):
    problem_statement = problem_statement + '.' if not problem_statement.endswith('.') else problem_statement
    examples = dict_to_str(examples_dict, False)
    definitions = get_definitions(['../samples-rich.csv', '../samples-v1.csv'])
    formal_definitions = dict_to_str(definitions, False)
    template = f'Rephrase the following geometry problem such that it is easy to formalize\n' \
               f'\'{problem_statement}\'\n' \
               f'Here are a few examples to show you how to rephrase.\n{examples}\n\n' \
               f'Notice that in the rephrased text every sentence correspond to one formal sentence defined below.\n' \
               f'{formal_definitions}\n\n' \
               f'Respond with just the rephrasing nothing else. Remember not to remove the capitalization of the ' \
               f'point names. Pay attention such that every rephrased statement is directly connected to one and ' \
               f'only one of the definitions.'

    return template


if __name__ == '__main__':
    # Example usage
    file_path = '../imo_ag_30.txt'  # Replace with your file path
    formal_descriptions = read_formal_descriptions(file_path)

    verbalizer = IndependentStatementVerbalization(None)

    # Print the list of formal descriptions
    num_exampes = 6
    example_dict = {}
    problem_statement = ''
    for i, (name, fl_statement) in enumerate(formal_descriptions.items()):
        name = name.replace('translated_', '')
        if name not in imo_problems.problems:
            continue
        try:
            nl_statement = verbalize_problem(verbalizer, fl_statement)
        except IndexError:
            continue

        if num_exampes > i:
            example_dict[i] = f'\nNatural: {imo_problems.problems[name]} \n' \
                              f'Rephrased: {nl_statement}'
        else:
            problem_statement = imo_problems.problems[name]
            break

    print(get_prompt(example_dict, problem_statement))