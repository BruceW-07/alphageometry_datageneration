import sys
sys.path.append('..')

from utils.loading_utils import load_definitions_and_rules
import random
import ddar
from alphageometry import write_solution
import graph as gh
import problem as pr
from random_generation.clause_generation import CompoundClauseGen
import signal
import copy
from draw_svg.get_svg import draw_svg
from cycleGAN.my_utils.point_naming_util import rename_point_names, reverse_mapping


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
    # defs_path = '../wrong_defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cc_gen = CompoundClauseGen(definitions, 2, 3, 2, 42,
                               shuffle_var_names=False)
    txt = cc_gen.generate_clauses()
    # txt = 'a b c d = rectangle a b c d; e = on_line e a c, on_line e b d'
    txt = ('a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; '
           'm = on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n g2 b; '
           'c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b, on_circle d g2 b; '
           'e = on_line e b d, on_line e a c; '
           'p = on_line p a n, on_line p c d; q = on_line q b n, on_line q c d ? cong e p e q')
    # txt = 'a b c = risos a b c'
    # txt = 'A B C X = eq_trapezoid X A B C'
    # txt = 'B = free B; A = free A; C = free C; D = orthocenter D A B C'
        # txt = 'A B C = risos A B C; D = eqangle2 D C A B; E = angle_mirror E A D C, angle_mirror E D A C; F = orthocenter F D B C; G H = on_pline G A E F, angle_mirror H E A B'
    # txt = 'A = free A; B = free B; C = free C; D = circle A B C'
    # Let P be an interior point of triangle ABC and AP, BP, CP meet the sides BC, CA, AB in D, E, F respectively. Show that AP/PD = AF/FB + AE/EC
    # txt = 'A = free A; B = free B; C = free C; X = circle X A B C; D = on_circle D X A; P = on_line P A C, on_line P B D; Q = on_tline Q P P C, on_line Q B C; C1 = circle C1 A P D; C2 = circle C2 B Q D ? para C1 C2 A D'
    # txt = 'b c = segment b c; o = midpoint o b c; a = on_circle a o b; d = on_circle d o b, on_bline d a b; e = on_bline e o a, on_circle e o b; f = on_bline f o a, on_circle f o b; j = on_pline j o a d, on_line j a c ? eqangle c e c j c j c f'
    # txt = 'A B C D = quadrangle A B C D; E F G H = incenter2 E F G H B C D; I = on_tline I B A D; J = angle_mirror J G C A, on_opline E G; K L M N = excenter2 K L M N A J G; O P Q R = r_trapezoid O P Q R; S T = on_pline S A C D, angle_bisector T R B G'

    org_2_alpha_geo, renamed_txt = rename_point_names(set(definitions.keys()), txt)
    alpha_geo_2_org = reverse_mapping(org_2_alpha_geo)
    print(f'alpha_geo_2_org:\n{org_2_alpha_geo}')

    txt, goal_str = txt.split('?')
    # goal_str = ''
    # print(txt)

    problem = pr.Problem.from_txt(txt.strip())

    print(f'Problem created, Building graph ...')
    try:
        # Set an alarm for 10 seconds
        # signal.alarm(1000)

        # Code block to execute with timeout
        graph, _ = gh.Graph.build_problem(problem, definitions)

        # Disable the alarm
        # signal.alarm(0)
    except TimeoutException as e:
        print("Graph couldn't be create in reasonable time. Perhaps problem with the premises. Exiting ...")
        raise e

    # import ipdb; ipdb.set_trace()
    # Additionaly draw this generated problem
    gh.nm.draw(
        graph.type2nodes[gh.Point],
        graph.type2nodes[gh.Line],
        graph.type2nodes[gh.Circle],
        graph.type2nodes[gh.Segment])

    svg_text = draw_svg(
        graph.type2nodes[gh.Point],
        graph.type2nodes[gh.Line],
        graph.type2nodes[gh.Circle],
        graph.type2nodes[gh.Segment],
        alpha_geo_2_org=alpha_geo_2_org,
        org_2_alpha_geo=org_2_alpha_geo,
        goal_str=goal_str.strip())

    # Write SVG code to a file
    with open('output.svg', 'w') as f:
        f.write(svg_text)

    print(f'Solving ...')
    exit(0)

    ddar.solve(graph, rules, problem, max_level=5)

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
