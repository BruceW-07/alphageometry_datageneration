import sys
sys.path.append('..')
import random
import ddar
from alphageometry import write_solution
import graph as gh
import problem as pr


def load_definitions_and_rules(defs_path, rules_path):
    """Load definitions and rules from text files."""
    definitions = pr.Definition.from_txt_file(defs_path, to_dict=True)
    rules = pr.Theorem.from_txt_file(rules_path, to_dict=True)
    return definitions, rules


def main():
    random.seed(0)
    # Example entities and conditions for illustration purposes

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    txt = 'a b c = triangle a b c; x = circle x a b c ? perp a b c x'
    # txt = 'a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c; f = midpoint f c a; g = on_line g d c, on_line g e a ? coll f b g'

    # A goal less starting point of the search
    txt = 'a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c; f = midpoint f c a; g = on_line g d c, on_line g e a'
    p = pr.Problem.from_txt(txt)
    g, _ = gh.Graph.build_problem(p, definitions)

    ddar.solve(g, rules, p, max_level=1000)

    # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
    cache_node = random.choice(list((g.cache.keys())))
    goal = pr.Construction(cache_node[0], list(cache_node[1:]))
    write_solution(g, p, goal=goal, out_file='')


if __name__ == "__main__":
    main()
