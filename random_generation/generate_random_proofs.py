import sys

from utils.loading_utils import load_definitions_and_rules

sys.path.append('..')
import random
import ddar
from alphageometry import write_solution
import graph as gh
import problem as pr
from clause_generation import CompoundClauseGen
import signal
import copy


def convert_var_names_from_alpha_geo_names(var_map, goal_statement_as_list):
    new_var_statement = copy.deepcopy(goal_statement_as_list)
    for var_id in range(1, len(goal_statement_as_list)):
        new_var_statement[var_id] = var_map[goal_statement_as_list[var_id]]

    return new_var_statement


class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)


def main():
    random.seed(4)
    # Example entities and conditions for illustration purposes

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cc_gen = CompoundClauseGen(definitions, 2, 3, 2, 42,
                               shuffle_var_names=False)
    txt = cc_gen.generate_clauses()
        # txt = 'A B C = risos A B C; D = eqangle2 D C A B; E = angle_mirror E A D C, angle_mirror E D A C; F = orthocenter F D B C; G H = on_pline G A E F, angle_mirror H E A B'
    # txt = 'A = free A; B = free B; C = free C; D = circle A B C'
    # Let P be an interior point of triangle ABC and AP, BP, CP meet the sides BC, CA, AB in D, E, F respectively. Show that AP/PD = AF/FB + AE/EC
    # txt = 'A = free A; B = free B; C = free C; X = circle X A B C; D = on_circle D X A; P = on_line P A C, on_line P B D; Q = on_tline Q P P C, on_line Q B C; C1 = circle C1 A P D; C2 = circle C2 B Q D ? para C1 C2 A D'
    # txt = 'b c = segment b c; o = midpoint o b c; a = on_circle a o b; d = on_circle d o b, on_bline d a b; e = on_bline e o a, on_circle e o b; f = on_bline f o a, on_circle f o b; j = on_pline j o a d, on_line j a c ? eqangle c e c j c j c f'
    # txt = 'A B C D = quadrangle A B C D; E F G H = incenter2 E F G H B C D; I = on_tline I B A D; J = angle_mirror J G C A, on_opline E G; K L M N = excenter2 K L M N A J G; O P Q R = r_trapezoid O P Q R; S T = on_pline S A C D, angle_bisector T R B G'

    # txt = txt.split('?')[0].strip()
    print(txt)

    problem = pr.Problem.from_txt(txt)

    print(f'Problem created, Building graph ...')
    try:
        # Set an alarm for 10 seconds
        signal.alarm(1000)

        # Code block to execute with timeout
        graph, _ = gh.Graph.build_problem(problem, definitions)

        # Disable the alarm
        signal.alarm(0)
    except TimeoutException as e:
        print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Exiting ...")
        raise e

    # # Additionaly draw this generated problem
    # gh.nm.draw(
    #     graph.type2nodes[gh.Point],
    #     graph.type2nodes[gh.Line],
    #     graph.type2nodes[gh.Circle],
    #     graph.type2nodes[gh.Segment])

    print(f'Solving ...')

    ddar.solve(graph, rules, problem, max_level=3)

    # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
    # random.seed(4)
    cache_node = list(random.choice(list((graph.cache.keys()))))
    # capitalized_pt_names = [point_name.capitalize() for point_name in cache_node[1:]]
    # cache_node[1:] = capitalized_pt_names
    var_map = cc_gen.get_varname_2_alpha_geo_var_map()
    # new_cache_node = convert_var_names_from_alpha_geo_names(var_map, cache_node)
    goal = pr.Construction(cache_node[0], list(cache_node[1:]))
    nl_soln, fl_soln = write_solution(graph, problem, goal=goal, out_file='')
    print(nl_soln)
    print(fl_soln)


if __name__ == "__main__":
    main()
