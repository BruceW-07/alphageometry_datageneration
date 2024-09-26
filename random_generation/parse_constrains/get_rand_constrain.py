from pretty import pretty_nl
import random


class ConstraintGenerator:
    def __init__(self, path_to_rules):
        self.constraints = self.parse_constraints(path_to_rules)

    def parse_constraints(self, path_to_rules):
        constraints = {}

        with open(path_to_rules, 'r') as file:
            for line in file:
                # Remove any leading/trailing whitespace
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Split the line at '=>'
                if '=>' in line:
                    lhs, rhs = line.split('=>')
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                else:
                    lhs = line
                    rhs = None

                # Process both lhs and rhs
                for part in [lhs, rhs]:
                    if part is None:
                        continue
                    # Split the constraints at ','
                    constraints_in_part = [cnst.strip() for cnst in part.split(',')]
                    for cnst in constraints_in_part:
                        # Split the constraint into name and arguments
                        cnst_components = cnst.split()
                        if len(cnst_components) >= 2:
                            cnst_name = cnst_components[0]
                            args = cnst_components[1:]
                            # Don't add goal constrains that are not pretty printable.
                            # most likely they are non-provable constraints anyway!
                            if pretty_nl(cnst_name, args) is None:
                                continue
                            # Create argument placeholders, keeping track of repeated arguments
                            arg_placeholders = []
                            arg_map = {}
                            arg_counter = 1
                            for arg in args:
                                if arg in arg_map:
                                    placeholder = arg_map[arg]
                                else:
                                    placeholder = f'{{arg{arg_counter}}}'
                                    arg_map[arg] = placeholder
                                    arg_counter += 1
                                arg_placeholders.append(placeholder)
                            # Create the template
                            template = cnst_name + ' ' + ' '.join(arg_placeholders)
                            num_unique_args = len(arg_map)
                            num_total_args = len(args)
                            # Store the template
                            if cnst_name not in constraints:
                                constraints[cnst_name] = []
                            template_entry = {
                                'template': template,
                                'num_args': num_total_args,
                                'num_unique_args': num_unique_args,
                                'arg_pattern': arg_placeholders,
                            }
                            # Avoid duplicates
                            if template_entry not in constraints[cnst_name]:
                                constraints[cnst_name].append(template_entry)
        return constraints

    def generate_constraint(self, points):
        # Collect all templates that can be instantiated with given points
        possible_templates = []
        for cnst_name, templates in self.constraints.items():
            for template_entry in templates:
                if template_entry['num_unique_args'] <= len(points):
                    possible_templates.append((cnst_name, template_entry))

        # If no possible templates, cannot proceed
        if not possible_templates:
            print("No constraints can be instantiated with the given points.")
            return None
        else:
            # Randomly select a template
            cnst_name, chosen_template = random.choice(possible_templates)
            num_unique_args = chosen_template['num_unique_args']

            # Randomly select num_unique_args point names
            selected_points = random.sample(points, num_unique_args)

            # Map the placeholders to selected point names
            placeholder_map = {}
            arg_counter = 1
            for point in selected_points:
                placeholder = f'{{arg{arg_counter}}}'
                placeholder_map[placeholder] = point
                arg_counter += 1

            # Build argument list, replacing placeholders with point names
            args = []
            for placeholder in chosen_template['arg_pattern']:
                args.append(placeholder_map[placeholder])

            # Instantiate the constraint
            instanciated_constraint = cnst_name + ' ' + ' '.join(args)
            return instanciated_constraint


# Example usage
if __name__ == "__main__":
    file_path = '../../rules.txt'
    generator = ConstraintGenerator(file_path)

    # Let's assume we have a list of point names
    points = ['A', 'B', 'C', 'D']

    # Generate a constraint
    constraint = generator.generate_constraint(points)
    if constraint:
        print(constraint)
