import sys
import re
from math import factorial
from itertools import permutations, combinations


class GeometryEquivalenceAnalyzer:
    """
    A class to analyze and determine equivalence between geometric figures.
    """

    def __init__(self):
        # Define all known geometric constructs
        self.definitions = [
            "angle_bisector", "angle_mirror", "circle", "circumcenter", "eq_quadrangle",
            "eq_trapezoid", "eq_triangle", "eqangle2", "eqdia_quadrangle", "eqdistance",
            "foot", "free", "incenter", "incenter2", "excenter", "excenter2",
            "centroid", "ninepoints", "intersection_cc", "intersection_lc",
            "intersection_ll", "intersection_lp", "intersection_lt", "intersection_pp",
            "intersection_tt", "iso_triangle", "lc_tangent", "midpoint", "mirror",
            "nsquare", "on_aline", "on_aline2", "on_bline", "on_circle", "on_line",
            "on_pline", "on_tline", "orthocenter", "parallelogram", "pentagon",
            "psquare", "quadrangle", "r_trapezoid", "r_triangle", "rectangle", "reflect",
            "risos", "s_angle", "segment", "shift", "square", "isquare", "trapezoid",
            "triangle", "triangle12", "2l1c", "e5128", "3peq", "trisect", "trisegment",
            "on_dia", "ieq_triangle", "on_opline", "cc_tangent0", "cc_tangent",
            "eqangle3", "tangent", "on_circum", "eqangle", "eqratio", "perp", "para", "cong",
            "cyclic", "coll", "midp"
        ]
        self.construct_to_id = {name: idx + 1 for idx, name in enumerate(self.definitions)}

        escaped_defs = [re.escape(d) for d in self.definitions]
        pattern = r'\b(' + '|'.join(escaped_defs) + r')\b'
        self.construct_pattern = re.compile(pattern)
        self.target_pattern = re.compile(pattern)
        self.geometry_map = {}

    def are_same_figure(self, figure1, figure2):
        """
        Determine whether two strings represent the same figure.
        """
        # Step 1: Parse the figure descriptions
        components1 = self.parse_figure(figure1)
        components2 = self.parse_figure(figure2)

        # Step 2: Check if they have the same number of components
        if len(components1) != len(components2):
            return False

        # Step 3: Try to find a mapping between points in figure1 and figure2
        point_mapping = self.find_point_mapping(components1, components2)

        # Step 4: If no valid mapping is found, they are not the same figure
        if point_mapping is None:
            return False

        # Step 5: Check if all components match under the point mapping
        return self.check_components_match(components1, components2, point_mapping)

    def parse_figure(self, figure_str):
        """Parse a figure string into its component parts."""
        # Split by semicolon to get individual components
        parts = figure_str.split(';')

        # Extract the question part if it exists
        question_part = None
        components = []

        for part in parts:
            if '?' in part:
                before_q, after_q = part.split('?', 1)
                components.append(before_q.strip())
                question_part = after_q.strip()
            else:
                components.append(part.strip())

        # Add the question part as a separate component if it exists
        if question_part:
            components.append('?' + question_part)

        # Parse each component into a structured format
        parsed_components = []
        for component in components:
            # Skip empty components
            if not component:
                continue

            # Parse the component based on its structure
            if '=' in component:
                left, right = component.split('=', 1)
                points = left.strip().split()

                # Split the right side by comma to handle multiple constraints
                constraints = [c.strip() for c in right.split(',')]

                for constraint in constraints:
                    construct_parts = constraint.split()
                    construct_type = construct_parts[0]
                    construct_args = construct_parts[1:]

                    parsed_components.append({
                        'type': 'definition',
                        'points': points,
                        'construct': construct_type,
                        'args': construct_args
                    })
            elif component.startswith('?'):
                # This is a query component
                query_type = component[1:].strip().split()[0]
                query_args = component[1:].strip().split()[1:]

                parsed_components.append({
                    'type': 'query',
                    'query': query_type,
                    'args': query_args
                })

        return parsed_components

    def extract_points(self, components):
        """Extract all unique points mentioned in the components."""
        points = set()
        for component in components:
            if component['type'] == 'definition':
                points.update(component['points'])
                points.update(component['args'])

        return points

    def find_unique_constructs(self, components1, components2):
        """Find construct types that appear exactly once in each figure."""
        constructs1 = [c.get('construct') for c in components1 if 'construct' in c]
        constructs2 = [c.get('construct') for c in components2 if 'construct' in c]

        # Find constructs that appear exactly once in both
        unique_constructs = []
        for construct in set(constructs1).intersection(set(constructs2)):
            if constructs1.count(construct) == 1 and constructs2.count(construct) == 1:
                unique_constructs.append(construct)

        # Sort by the number of points they define (fewer is better for starting)
        unique_constructs.sort(key=lambda c: self.EQUIVALENT_COUNTS.get(c, 1))

        return unique_constructs

    CONSTRUCT_EQUIVALENCES = [
        # 基本构造类型
        ('angle_bisector', 'x a b c', lambda args: [
            args,  # 原始顺序: x a b c
            [args[0], args[2], args[1], args[3]]  # a<->c交换: x c b a
        ]),

        ('circle', 'x a b c', lambda args: [
            [args[0]] + list(p) for p in permutations(args[1:4])  # a,b,c的所有排列(3! = 6种)
        ]),

        ('circumcenter', 'x a b c', lambda args: [
            [args[0]] + list(p) for p in permutations(args[1:4])  # 同circle
        ]),

        ('eq_triangle', 'x b c', lambda args: [
            args,  # x b c
            [args[0], args[2], args[1]]  # x c b
        ]),

        ('eqangle2', 'x a b c', lambda args: [
            args,  # x a b c
            [args[0], args[2], args[1], args[3]]  # x c b a
        ]),

        ('incenter', 'x a b c', lambda args: [
            [args[0]] + list(p) for p in permutations(args[1:4])  # a,b,c的所有排列
        ]),

        # 特殊构造类型（轮换等价）
        ('incenter2', 'x y z i a b c', lambda args: [
            args,  # x y z i a b c
            [args[1], args[2], args[0], args[3], args[5], args[6], args[4]],  # y z x i b c a
            [args[2], args[0], args[1], args[3], args[6], args[4], args[5]]  # z x y i c a b
        ]),

        ('excenter', 'x a b c', lambda args: [
            [args[0]] + list(p) for p in permutations(args[1:4])  # 同incenter
        ]),

        ('excenter2', 'x y z i a b c', lambda args: [
            args,  # x y z i a b c
            [args[1], args[2], args[0], args[3], args[5], args[6], args[4]],  # y z x i b c a
            [args[2], args[0], args[1], args[3], args[6], args[4], args[5]]  # z x y i c a b
        ]),

        ('centroid', 'x y z i a b c', lambda args: [
            args,  # x y z i a b c
            [args[1], args[2], args[0], args[3], args[5], args[6], args[4]],  # y z x i b c a
            [args[2], args[0], args[1], args[3], args[6], args[4], args[5]]  # z x y i c a b
        ]),

        ('ninepoints', 'x y z i a b c', lambda args: [
            args,  # x y z i a b c
            [args[1], args[2], args[0], args[3], args[5], args[6], args[4]],  # y z x i b c a
            [args[2], args[0], args[1], args[3], args[6], args[4], args[5]]  # z x y i c a b
        ]),

        # 三角形相关
        ('iso_triangle', 'a b c', lambda args: [
            args,  # a b c
            [args[0], args[2], args[1]]  # a c b
        ]),

        ('ieq_triangle', 'a b c', lambda args: [
            list(p) for p in permutations(args)  # 所有排列(3! = 6种)
        ]),

        ('triangle', 'a b c', lambda args: [
            list(p) for p in permutations(args)  # 所有排列
        ]),

        ('r_triangle', 'a b c', lambda args: [
            args,  # a b c
            [args[0], args[2], args[1]]  # a c b
        ]),

        # 中点相关
        ('midpoint', 'x a b', lambda args: [
            args,  # x a b
            [args[0], args[2], args[1]]  # x b a
        ]),

        ('midp', 'a b c', lambda args: [
            args,  # a b c
            [args[0], args[2], args[1]]  # a c b
        ]),

        # 距离和线段相关
        ('eqdistance', 'x a b c', lambda args: [
            args,  # x a b c
            [args[0], args[1], args[3], args[2]]  # x a c b
        ]),

        ('segment', 'a b', lambda args: [
            args,  # a b
            [args[1], args[0]]  # b a
        ]),

        # 共线相关
        ('coll', '*points', lambda args: [
            list(p) for p in permutations(args)  # 所有点的排列(n!种)
        ]),

        # 四边形相关
        ('parallelogram', 'a b c x', lambda args: [
            args,  # a b c x
            [args[2], args[1], args[0], args[3]]  # c b a x
        ]),

        ('square', 'a b x y', lambda args: [
            args,  # a b x y
            [args[1], args[0], args[3], args[2]]  # b a y x
        ]),

        ('rectangle', 'a b c d', lambda args: [
            [args[i], args[(i + 1) % 4], args[(i + 2) % 4], args[(i + 3) % 4]] for i in range(4)  # 循环排列
        ]),

        # 其他构造类型
        ('cyclic', 'a b c d', lambda args: [
            list(p) for p in permutations(args)  # 所有排列(4! = 24种)
        ]),

        ('perp', 'a b c d', lambda args: [
            args,  # a b c d
            [args[1], args[0], args[2], args[3]],  # b a c d
            [args[0], args[1], args[3], args[2]],  # a b d c
            [args[1], args[0], args[3], args[2]]  # b a d c
        ]),

    ]

    EQUIVALENT_COUNTS = {}
    for item in CONSTRUCT_EQUIVALENCES:
        sample_args = item[1].split() if item[1] != '*points' else ['a','b','c']
        EQUIVALENT_COUNTS[item[0]] = len(item[2](sample_args))

    def update_mapping_for_construct(self, comp1, comp2, current_mapping):
        """
        Update the point mapping based on a pair of components with the same construct type.
        Takes into account the equivalence rules for different construct types using CONSTRUCT_EQUIVALENCES.
        """
        construct_type = comp1.get('construct')

        # Find the equivalence rule for this construct type
        equiv_rule = None
        param_format = None
        for rule_type, params, rule_func in self.CONSTRUCT_EQUIVALENCES:
            if rule_type == construct_type:
                equiv_rule = rule_func
                param_format = params
                break

        # If no equivalence rule is found, use direct mapping
        if equiv_rule is None:
            return self._direct_mapping(comp1, comp2, current_mapping)

        # Special handling for variable number of points
        if param_format == '*points':
            points1 = comp1.get('points', []) + comp1.get('args', [])
            points2 = comp2.get('points', []) + comp2.get('args', [])

            if len(points1) != len(points2):
                return None

            # Try each possible permutation from the equivalence rules
            for perm in equiv_rule(points1):
                # Try to create a mapping using this permutation
                new_mapping = current_mapping.copy()
                valid = True

                for i, p1 in enumerate(points1):
                    if i < len(perm):
                        p2_idx = points1.index(perm[i])
                        p2 = points2[p2_idx] if p2_idx < len(points2) else None

                        if p2 is None or (p1 in new_mapping and new_mapping[p1] != p2):
                            valid = False
                            break
                        new_mapping[p1] = p2

                if valid:
                    return new_mapping

            return None

        # Normal case: extract points based on parameter format
        param_list = param_format.split()

        # Gather the points from comp1 and comp2
        points1 = []
        points2 = []

        # First try to use 'args' if available
        if 'args' in comp1 and 'args' in comp2:
            points1 = comp1.get('args', [])
            points2 = comp2.get('args', [])
        # Otherwise use 'points'
        elif 'points' in comp1 and 'points' in comp2:
            points1 = comp1.get('points', [])
            points2 = comp2.get('points', [])
        # Or combine both
        else:
            points1 = comp1.get('points', []) + comp1.get('args', [])
            points2 = comp2.get('points', []) + comp2.get('args', [])

        # Check that we have the right number of points
        if len(points1) != len(param_list) or len(points2) != len(param_list):
            return None

        # Generate all possible equivalent orderings
        equivalent_orderings = equiv_rule(points1)

        # Try each ordering to find a valid mapping
        for ordering in equivalent_orderings:
            new_mapping = current_mapping.copy()
            valid = True

            for i, p1 in enumerate(ordering):
                if i < len(points2):
                    p2 = points2[i]
                    if p1 in new_mapping and new_mapping[p1] != p2:
                        valid = False
                        break
                    new_mapping[p1] = p2

            if valid:
                return new_mapping

        return None

    def _direct_mapping(self, comp1, comp2, current_mapping):
        """
        Helper method for direct mapping when no specific equivalence rule exists.
        """
        new_mapping = current_mapping.copy()

        # Map points based on their positions
        if 'points' in comp1 and 'points' in comp2:
            if len(comp1['points']) != len(comp2['points']):
                return None

            for i, p1 in enumerate(comp1['points']):
                p2 = comp2['points'][i]
                if p1 in new_mapping and new_mapping[p1] != p2:
                    return None
                new_mapping[p1] = p2

        if 'args' in comp1 and 'args' in comp2:
            if len(comp1['args']) != len(comp2['args']):
                return None

            for i, p1 in enumerate(comp1['args']):
                p2 = comp2['args'][i]
                if p1 in new_mapping and new_mapping[p1] != p2:
                    return None
                new_mapping[p1] = p2

        return new_mapping

    def find_point_mapping(self, components1, components2):
        """
        Find a valid mapping between points in figure1 and figure2.
        """
        # Extract all unique points from both figures
        points1 = self.extract_points(components1)
        points2 = self.extract_points(components2)

        # If they have different numbers of points, they can't be the same figure
        if len(points1) != len(points2):
            return None

        # Start with unique constructs (those that appear only once)
        unique_constructs = self.find_unique_constructs(components1, components2)

        # Initialize the point mapping
        point_mapping = {}

        # Use the unique constructs to establish initial point mappings
        for construct_type in unique_constructs:
            # Find the components with this construct type
            comp1 = next((c for c in components1 if c.get('construct') == construct_type), None)
            comp2 = next((c for c in components2 if c.get('construct') == construct_type), None)

            if comp1 and comp2:
                # Update the point mapping based on the equivalence rules for this construct
                updated_mapping = self.update_mapping_for_construct(comp1, comp2, point_mapping)

                # If we couldn't find a consistent mapping, this might not be the same figure
                if updated_mapping is None:
                    return None

                point_mapping = updated_mapping

        # If we haven't mapped all points, try to infer the remaining mappings
        if len(point_mapping) < len(points1):
            # Use the remaining constructs to fill in the mapping
            point_mapping = self.complete_mapping(components1, components2, point_mapping, points1, points2)

        # Check if the mapping is complete and consistent
        if len(point_mapping) == len(points1) and self.is_mapping_consistent(components1, components2, point_mapping):
            return point_mapping

        return None

    def complete_mapping(self, components1, components2, partial_mapping, points1, points2):
        """
        Try to complete the partial mapping using the remaining components.
        """
        # Create a copy of the partial mapping
        mapping = partial_mapping.copy()

        # Identify unmapped points
        unmapped_points1 = points1 - set(mapping.keys())
        unmapped_points2 = points2 - set(mapping.values())

        # If no more points to map, return the current mapping
        if not unmapped_points1:
            return mapping

        # Try all possible assignments for the first unmapped point
        p1 = next(iter(unmapped_points1))

        for p2 in unmapped_points2:
            test_mapping = mapping.copy()
            test_mapping[p1] = p2

            # Check if this mapping is consistent with all components
            if self.is_mapping_consistent(components1, components2, test_mapping):
                # Recursively try to complete the mapping
                remaining_unmapped1 = unmapped_points1 - {p1}
                remaining_unmapped2 = unmapped_points2 - {p2}

                if not remaining_unmapped1:
                    return test_mapping

                complete_test_mapping = test_mapping.copy()
                # Pair remaining unmapped points in order (simplified approach)
                for r1, r2 in zip(remaining_unmapped1, remaining_unmapped2):
                    complete_test_mapping[r1] = r2

                if self.is_mapping_consistent(components1, components2, complete_test_mapping):
                    return complete_test_mapping

        # If no valid complete mapping is found
        return mapping

    def angles_match(self, angle1, angle2):
        """
        Check if two angles match. Angles are represented as [a, b, c, d] for angle between lines ab and cd.
        """
        if len(angle1) != 4 or len(angle2) != 4:
            return False

        # Direct match
        if angle1 == angle2:
            return True

        # Reversed first line
        if [angle1[1], angle1[0], angle1[2], angle1[3]] == angle2:
            return True

        # Reversed second line
        if [angle1[0], angle1[1], angle1[3], angle1[2]] == angle2:
            return True

        # Both lines reversed
        if [angle1[1], angle1[0], angle1[3], angle1[2]] == angle2:
            return True

        # Swapped lines
        if [angle1[2], angle1[3], angle1[0], angle1[1]] == angle2:
            return True

        # Swapped and first line reversed
        if [angle1[3], angle1[2], angle1[0], angle1[1]] == angle2:
            return True

        # Swapped and second line reversed
        if [angle1[2], angle1[3], angle1[1], angle1[0]] == angle2:
            return True

        # Swapped and both lines reversed
        if [angle1[3], angle1[2], angle1[1], angle1[0]] == angle2:
            return True

        return False

    def is_args_equivalent(self, construct_type, args1, args2):
        """
        Check if two sets of arguments are equivalent based on the construct type.
        """
        if len(args1) != len(args2):
            return False

        # Handle specific construct types and their equivalence rules
        if construct_type == 'angle_bisector':
            # angle_bisector x a b c | a and c are equivalent
            if args1[0] != args2[0]:
                return False

            if (args1[1] == args2[1] and args1[2] == args2[2] and args1[3] == args2[3]) or \
                    (args1[1] == args2[3] and args1[2] == args2[2] and args1[3] == args2[1]):
                return True

            return False

        elif construct_type == 'circle':
            # circle x a b c | a, b, c are equivalent (three points on the circle)
            if args1[0] != args2[0]:
                return False

            return set(args1[1:4]) == set(args2[1:4])

        elif construct_type == 'on_tline':
            # on_tline x a b c | b c are equivalent
            if args1[0] != args2[0]:
                return False

            if (args1[1] == args2[1] and (args1[2] == args2[2] or args1[2] == args2[3])) or \
                    (args1[1] == args2[2] and (args1[2] == args2[1] or args1[2] == args2[3])) or \
                    (args1[1] == args2[3] and (args1[2] == args2[1] or args1[2] == args2[2])):
                return True

            return False

        elif construct_type == 'tangent':
            # tangent x y a o b | x y are equivalent
            if len(args1) < 2 or len(args2) < 2:
                return False

            # Check if the first two points match (can be swapped)
            if (args1[0] == args2[0] and args1[1] == args2[1]) or \
                    (args1[0] == args2[1] and args1[1] == args2[0]):
                # Check remaining points
                remaining_match = True
                for i in range(2, len(args1)):
                    if i < len(args2) and args1[i] != args2[i]:
                        remaining_match = False
                        break

                if remaining_match:
                    return True

            return False

        elif construct_type == 'eqangle':
            # eqangle a b c d e f g h | Angles can be compared
            if len(args1) != 8 or len(args2) != 8:
                return False

            # Extract the angles
            angle1_1 = args1[0:4]  # a b c d
            angle1_2 = args1[4:8]  # e f g h
            angle2_1 = args2[0:4]
            angle2_2 = args2[4:8]

            # Check if the angles match (can be swapped or reversed)
            if self.angles_match(angle1_1, angle2_1) and self.angles_match(angle1_2, angle2_2):
                return True

            if self.angles_match(angle1_1, angle2_2) and self.angles_match(angle1_2, angle2_1):
                return True

            return False

        # Default case: check if the arguments are exactly the same
        return args1 == args2

    def is_mapping_consistent(self, components1, components2, mapping):
        """
        Check if the given mapping is consistent with all components.
        """
        # Apply the mapping to the first figure and check if it matches the second figure
        transformed_components = []

        for comp1 in components1:
            if comp1['type'] == 'definition':
                transformed_points = [mapping.get(p, p) for p in comp1['points']]
                transformed_args = [mapping.get(p, p) for p in comp1['args']]

                transformed_comp = {
                    'type': 'definition',
                    'points': transformed_points,
                    'construct': comp1['construct'],
                    'args': transformed_args
                }
                transformed_components.append(transformed_comp)

            elif comp1['type'] == 'query':
                transformed_args = [mapping.get(p, p) for p in comp1['args']]

                transformed_comp = {
                    'type': 'query',
                    'query': comp1['query'],
                    'args': transformed_args
                }
                transformed_components.append(transformed_comp)

        # Check if each transformed component matches a component in figure2
        for trans_comp in transformed_components:
            found_match = False

            for comp2 in components2:
                if trans_comp['type'] != comp2['type']:
                    continue

                if trans_comp['type'] == 'definition':
                    if (trans_comp['construct'] == comp2['construct'] and
                            set(trans_comp['points']) == set(comp2['points']) and
                            self.is_args_equivalent(trans_comp['construct'], trans_comp['args'], comp2['args'])):
                        found_match = True
                        break

                elif trans_comp['type'] == 'query':
                    if (trans_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(trans_comp['query'], trans_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        return True


    def check_components_match(self, components1, components2, point_mapping):
        """
        Check if all components in figure1 match components in figure2 under the given point mapping.
        """
        # Apply the mapping to components1
        mapped_components = []

        for comp in components1:
            mapped_comp = self.map_component(comp, point_mapping)
            mapped_components.append(mapped_comp)

        # Check if each mapped component matches a component in figure2
        for mapped_comp in mapped_components:
            found_match = False

            for comp2 in components2:
                if mapped_comp['type'] != comp2['type']:
                    continue

                if mapped_comp['type'] == 'definition':
                    if (mapped_comp['construct'] == comp2['construct'] and
                            sorted(mapped_comp['points']) == sorted(comp2['points']) and
                            self.is_args_equivalent(mapped_comp['construct'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break

                elif mapped_comp['type'] == 'query':
                    if (mapped_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(mapped_comp['query'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        # Also check in reverse - every component in figure2 should match a component in mapped_components
        for comp2 in components2:
            found_match = False

            for mapped_comp in mapped_components:
                if mapped_comp['type'] != comp2['type']:
                    continue

                if mapped_comp['type'] == 'definition':
                    if (mapped_comp['construct'] == comp2['construct'] and
                            sorted(mapped_comp['points']) == sorted(comp2['points']) and
                            self.is_args_equivalent(mapped_comp['construct'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break


                elif mapped_comp['type'] == 'query':
                    if (mapped_comp['query'] == comp2['query'] and
                            self.is_args_equivalent(mapped_comp['query'], mapped_comp['args'], comp2['args'])):
                        found_match = True
                        break

            if not found_match:
                return False

        return True

    def map_component(self, component, point_mapping):
        """Map the points in a component according to the given point mapping."""
        mapped_component = component.copy()

        if component['type'] == 'definition':
            mapped_component['points'] = [point_mapping.get(p, p) for p in component['points']]
            mapped_component['args'] = [point_mapping.get(p, p) for p in component['args']]

        elif component['type'] == 'query':
            mapped_component['args'] = [point_mapping.get(p, p) for p in component['args']]

        return mapped_component

    def process_geometry_block(self, data_num, content):
        """Process a single geometry block from the input file."""
        if '?' not in content:
            print(f"⚠️ Data {data_num} missing question mark, skipping")
            return

        before_q, after_q = content.split('?', 1)
        # Extract constructs using regex
        constructs = re.findall(self.construct_pattern, before_q)
        construct_ids = sorted(self.construct_to_id[c] for c in constructs if c in self.construct_to_id)

        # Extract target construct
        target_match = re.search(self.target_pattern, '?' + after_q)
        if not target_match:
            print(f"⚠️ Data {data_num} missing target construct, skipping")
            return

        target_keyword = target_match.group(1)
        if target_keyword not in self.construct_to_id:
            print(f"⚠️ Data {data_num}'s target construct {target_keyword} not in definitions")
            return

        target_id = self.construct_to_id[target_keyword]

        # Store the structure and geometry
        key = (tuple(construct_ids), target_id)
        if key not in self.structure_map:
            self.structure_map[key] = []
        self.structure_map[key].append(data_num)
        self.geometry_map[data_num] = content

    def analyze_input_file(self, input_file):
        """Read and analyze geometry figures from the input file."""
        from collections import defaultdict
        self.structure_map = defaultdict(list)
        self.geometry_map = {}

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            i = 0
            while i < len(lines):
                if lines[i].isdigit():
                    data_num = int(lines[i])
                    i += 1
                    if i < len(lines):
                        content = lines[i]
                        self.process_geometry_block(data_num, content)
                i += 1

            return True
        except Exception as e:
            print(f"Error reading input file: {e}")
            return False

    def find_equivalent_figures(self):
        """Find all pairs of equivalent figures."""
        from itertools import combinations

        equivalent_pairs = []

        # For each structure group, check pairs within the group
        for structure_key, data_nums in self.structure_map.items():
            if len(data_nums) > 1:
                for a, b in combinations(data_nums, 2):
                    fig1 = self.geometry_map[a]
                    fig2 = self.geometry_map[b]
                    if self.are_same_figure(fig1, fig2):
                        equivalent_pairs.append((a, b))

        return equivalent_pairs

    def write_results_to_file(self, output_file, equivalent_pairs):
        """Write the results to the output file."""
        try:
            with open(output_file, "w", encoding="utf-8") as out_file:
                out_file.write("The following figure pairs are equivalent:\n\n")
                for a, b in equivalent_pairs:
                    out_file.write(f"{a} and {b} are equivalent figures\n")
            return True
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False

    def run_analysis(self, input_file="input.txt", output_file="output.txt"):
        """Run the complete analysis process."""
        print(f"Reading from {input_file}...")
        if not self.analyze_input_file(input_file):
            return False

        print("Finding equivalent figures...")
        equivalent_pairs = self.find_equivalent_figures()

        print(f"Writing results to {output_file}...")
        if not self.write_results_to_file(output_file, equivalent_pairs):
            return False

        print(f"Analysis complete. Found {len(equivalent_pairs)} equivalent figure pairs.")
        return True

    # Add this code at the bottom of your script
if __name__ == "__main__":

    # Parse command-line arguments
    input_filename = "input.txt"  # Default
    output_filename = "output.txt"  # Default

    # If an argument is provided, use it as the input filename
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]

    # Optional: Allow for output filename to be specified as second argument
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]

    # Assuming this function is a method of a class named something like Analyzer
    # You'll need to instantiate the class first, then call the method
    analyzer = GeometryEquivalenceAnalyzer()
    analyzer.run_analysis(input_filename, output_filename)
