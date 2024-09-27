import re

def rename_point_names(problem_description):
    """
    Renames point names in a formal geometry problem description according to predetermined names.

    Parameters:
    - problem_description (str): The original problem description.

    Returns:
    - renamed_problem_description (str): The problem description with point names renamed.
    """
    # List of function names and reserved words
    function_names = {'segment', 'on_tline', 'on_circle', 'on_pline', 'on_line', 'cong'}
    operators = {'=', ',', ';', '?'}
    reserved_words = function_names.union(operators)

    # Generate predetermined names: A-Z, A0-Z0, A1-Z1, ..., A9-Z9
    predetermined_names = []
    for suffix in ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        for c in map(chr, range(ord('A'), ord('Z')+1)):
            predetermined_names.append(c + suffix)

    # Step 1: Extract point names
    point_names_ordered = []
    statements = re.split(r'[;?]', problem_description)

    for statement in statements:
        if '=' in statement:
            _, right_side = statement.split('=', 1)
        else:
            right_side = statement
        tokens = re.findall(r'\b\w+\b', right_side)
        for token in tokens:
            if token not in reserved_words and not token.isdigit():
                if token not in point_names_ordered:
                    point_names_ordered.append(token)

    # Step 2: Create mapping from original point names to predetermined names
    mapping = {}
    for original_name, new_name in zip(point_names_ordered, predetermined_names):
        mapping[original_name] = new_name

    # Step 3: Replace point names in the problem description
    def replace_point_names(match):
        word = match.group(0)
        return mapping.get(word, word)

    pattern = re.compile(r'\b(' + '|'.join(re.escape(name) for name in point_names_ordered) + r')\b')
    renamed_problem_description = pattern.sub(replace_point_names, problem_description)

    return mapping, renamed_problem_description


def reverse_mapping(mapping):
    rev_map = {}
    for key, val in mapping.items():
        rev_map[val] = key
    return rev_map


# Example usage:
if __name__ == "__main__":
    problem_description = ('a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m = on_circle m g1 a, '
                           'on_circle m g2 b; n = on_circle n g1 a, on_circle n g2 b; c = on_pline c m a b, '
                           'on_circle c g1 a; d = on_pline d m a b, on_circle d g2 b; e = on_line e a c, '
                           'on_line e b d; p = on_line p a n, on_line p c d; q = on_line q b n, on_line q c d ? '
                           'cong e p e q')

    mapping, renamed_problem = rename_point_names(problem_description)
    print("Original problem description:")
    print(problem_description)
    print("\nRenamed problem description:")
    print(renamed_problem)
    print("\n mapping")
    print(mapping)
