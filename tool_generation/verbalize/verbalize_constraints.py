import yaml
import random
import string


def get_cnstr_name_args(constraint):
    name_and_args = constraint.split(' ')
    name, args = name_and_args[0], name_and_args[1:]
    return name.strip(), args

def get_constraint_from_yaml(fl_prob_yml):
    # Split the input into lines and strip whitespace
    lines = fl_prob_yml.strip().split('\n')

    # Remove the first line if it starts with triple backticks
    if lines[0].strip().startswith('```'):
        lines = lines[1:]  # Remove the first line
    if lines[-1].strip().endswith('```'):
        lines = lines[:-1]  # Remove the last line if it ends with triple backticks

    # Join the remaining lines back into a single string
    cleaned_yml = '\n'.join(lines).strip()

    try:
        parsed_probs = yaml.safe_load(cleaned_yml)
        if isinstance(parsed_probs, list):
            parsed_probs = parsed_probs[0]
        constraints = parsed_probs['problem']['constraints']
        return constraints

    except IndexError:
        return False


def extract_constraint_line(fl_prob_yml, constraint):
    # Find the position where 'name' starts
    start_pos = fl_prob_yml.find(constraint)

    if start_pos != -1:  # Ensure 'name' is found in the string
        # Find the position of the next newline character after 'name'
        end_pos = fl_prob_yml.find('\n', start_pos)

        if end_pos == -1:
            # If there's no newline, copy everything till the end of the string
            result = fl_prob_yml[start_pos:]
        else:
            # Copy the part from 'name' to the newline character
            result = fl_prob_yml[start_pos:end_pos]

        return result
    else:
        raise ValueError(f"The string {constraint} was not found.")


class PreDefConstVerbalizer:
    def __init__(self, pre_def_yml_path):
        with open(pre_def_yml_path, 'r') as file:
            self.yaml_content = file.read()

        self._constraint_dict = {}
        self.comment_like = self.verbalize_all_constraints(self.yaml_content, format='comment_like')
        self.current_fl_prob = None
        self.latest_error = None
        self.latest_new_symbols = None

    def get_comment_like(self):
        return self.comment_like

    # Function to generate random point names
    def generate_random_points(self, n):
        return [random.choice(string.ascii_uppercase) for _ in range(n)]

    # Function to verbalize a constraint using the new template and random points
    def verbalize_constraint(self, template, points):
        template_filled = template
        for i, point in enumerate(points):
            template_filled = template_filled.replace(f"{{arg{i}}}", point)
        return template_filled

    def validate_symbols_and_syntax(self, constraints):
        error_msgs = []
        unknown_constraints = []
        for constraint in constraints:
            name, args = get_cnstr_name_args(constraint)
            if not name in self._constraint_dict:
                unknown_constraints.append(name)
            elif len(args) != self._constraint_dict[name]['nargs']:
                error_msgs.append(f'Error: {name} takes {self._constraint_dict[name]["nargs"]} args {len(args)} '
                                  f'provided')

        if not error_msgs and not unknown_constraints:
            return 'Valid'
        elif error_msgs:
            return '\n'.join(error_msgs)
        else:
            return unknown_constraints

    def is_new_symbols_and_valid_syntax(self, fl_prob_yml):
        constraints = get_constraint_from_yaml(fl_prob_yml)
        if constraints:
            unknown_constraints = self.validate_symbols_and_syntax(constraints)
            self.latest_new_symbols = []
            for unk_cnstr in unknown_constraints:
                constraint_with_comment = extract_constraint_line(fl_prob_yml, unk_cnstr)
                if constraint_with_comment.find('#') < 0:
                    constraint_with_comment += f' # {constraint_with_comment}'
                self.latest_new_symbols.append(constraint_with_comment)
            if isinstance(unknown_constraints, list):
                return True
            else:
                return False
        else:
            return False

    def add_new_symbol(self, fl_def_new_sym):
        try:
            new_sym_dict = yaml.safe_load(fl_def_new_sym)
            if isinstance(new_sym_dict, list):
                new_sym_dict = new_sym_dict[0]
            new_sym_inst = new_sym_dict['problem']['usage']
            new_sym_name_and_args = new_sym_inst.split(' ')
            name, args = new_sym_name_and_args[0], new_sym_name_and_args[1:]
            self._constraint_dict[name] = {'template': new_sym_dict['problem']['comment_templates'], 'nargs': len(args)}
            return True
        except IndexError:
            return False

    def get_new_symbols(self):
        if self.latest_new_symbols is None:
            raise ValueError(f' Call "is_new_symbols_and_valid_syntax" first')
        else:
            latest_new_symbols = self.latest_new_symbols
            self.latest_new_symbols = None
            return latest_new_symbols

    def is_valid_symbol_valid_syntax(self, fl_prob_yml):
        constraints = get_constraint_from_yaml(fl_prob_yml)
        if constraints:
            is_valid = self.validate_symbols_and_syntax(constraints)
            if is_valid != 'Valid':
                return False

            self.current_fl_prob = constraints
            return True
        else:
            return False

    def get_latest_error_message(self):
        if self.latest_error is None:
            raise ValueError(f'Call first "is_valid_symbol_incorrect_syntax"')
        else:
            err_msg = self.latest_error
            self.latest_error = None
            return err_msg


    def is_valid_symbol_incorrect_syntax(self, fl_prob_yml):
        constraints = get_constraint_from_yaml(fl_prob_yml)
        if constraints:
            error_messages = self.validate_symbols_and_syntax(constraints)
            if isinstance(error_messages, str):
                self.latest_error = error_messages
                return True
            else:
                return False
        else:
            return False

    def get_verbalization_of_latest_fl_prob(self):
        if self.current_fl_prob is not None:
            verbalized = []
            for constraint in  self.current_fl_prob:
                name, args = get_cnstr_name_args(constraint)
                if isinstance(self._constraint_dict[name]['template'], list):
                    chosen_template = random.choice(self._constraint_dict[name]['template'])
                else:
                    chosen_template = self._constraint_dict[name]['template']
                verbalized.append(self.verbalize_constraint(chosen_template, args))
            self.current_fl_prob = None
            return ' '.join(verbalized)
        else:
            raise ValueError(f'self.current_fl_prob is None, call "is_valid_symbol_valid_syntax" first')

    def verbalize_constraint_dict(self, constraints, format):
        verbalized_yml = []
        # Iterate through constraints
        for constraint in constraints:
            name = constraint['name']
            comment = constraint['comment']
            arguments = constraint['arguments']
            template = random.choice(constraint['verbalization_templates'])
            self._constraint_dict[name] = {'template': template, 'nargs': arguments}

            # Generate random points
            points = self.generate_random_points(arguments)

            # Verbalize the constraint
            verbalized_text = self.verbalize_constraint(template, points)

            # Output the final result (Comment-like format)
            if format.lower() == 'comment_like':
                constraint_instantiation = f"{name} {' '.join(points)}"
                spaces = ' ' * (21 - len(constraint_instantiation))
                verbalized_yml.append(f"{constraint_instantiation}{spaces}# {verbalized_text}\n")
            elif format.lower() == 'problem_like':
                verbalized_yml.append(f'{verbalized_text} ')
        return ''.join(verbalized_yml)

    # Function to read YAML and process constraints
    def verbalize_all_constraints(self, yaml_content, format):
        # Load YAML content into a Python object
        constraints = yaml.safe_load(yaml_content)

        return self.verbalize_constraint_dict(constraints, format)

    def verbalize_instantiated(self, list_constraints):
        inst_verb = []
        resp_dict = {}
        for constraint in list_constraints:
            name, args = get_cnstr_name_args(constraint)
            if name not in self._constraint_dict:
                resp_dict['UnknownConstraint'] = {'name': name, 'arguments': len(args), 'comment': constraint,
                                                  'verbalization_templates': '-'}
                break
            elif len(args) != self._constraint_dict[name]['nargs']:
                resp_dict['Error'] = (f'Error: {name} takes {self._constraint_dict[name]["nargs"]} '
                                      f'arguments {len(args)} provided.')
                break
            else:
                inst_verb.append(self.verbalize_constraint(self._constraint_dict[name]['template'], args))

        resp_dict['verb'] = ' '.join(inst_verb)

        return resp_dict



if __name__ == '__main__':
    # Main execution
    yaml_file = '../prompt_gen/atomic_constraints.yml'
    verbelizer = PreDefConstVerbalizer(yaml_file)

    # print(verbalize_all_constraints(yaml_content, format='comment_like'))
    print(verbelizer.get_comment_like())

    constraints = ['free G1', 'free G2', 'free M', 'free N', 'cyclic G1 M N G2']
    print(verbelizer.verbalize_instantiated(constraints))
