import sys
sys.path.append('..')
import traceback

from absl import app
from absl import flags
from absl import logging
import alphageometry as ag
import ddar
import graph as gh
import pretty as pt
import problem as pr

SHAVE_PROBLEMS_FILE = flags.DEFINE_string(
    'shave_problems_file',
    'imo_ag_30.txt',
    'text file contains the problem strings. See imo_ag_30.txt for example.',
)
SHAVE_PROBLEM_NAME = flags.DEFINE_string(
    'shave_problem_name',
    'imo_2000_p1',
    'name of the problem to solve, must be in the problem_file.',
)

SHAVE_DEFS_FILE = flags.DEFINE_string(
    'shave_defs_file',
    'defs.txt',
    'definitions of available constructions to state a problem.',
)
SHAVE_RULES_FILE = flags.DEFINE_string(
    'shave_rules_file', 'rules.txt', 'list of deduction rules used by DD.'
)

SHAVE_OUT_FILE = flags.DEFINE_string(
    'shave_out_file', '', 'path to the solution output file.'
)  # pylint: disable=line-too-long

def pretty(con, delete_point=False, to_upper=False):
    points, con_str = con.split(' = ')
    points = points.split(' ')
    con_str = con_str.split(' ')
    if to_upper:
        points = [p.capitalize() for p in points]
        con_str = [con_str[i].capitalize() if i > 0 else con_str[i] for i in range(len(con_str))]
    else :
        points = [p for p in points]
        con_str = [con_str[i] if i > 0 else con_str[i] for i in range(len(con_str))]
    return ' '.join(points) + ' = ' + ' '.join(con_str) if not delete_point else ' '.join(con_str)

def find_essential_clauses(g, pr):
    essential_clauses, essential_aux_clauses = ddar.get_essential_clauses(g, pr.goal)
    statement = []
    for clause in pr.clauses:
        if clause.txt() in essential_clauses or clause.txt() in essential_aux_clauses:
            statement.append(clause.txt())
    return '; '.join(statement) + ' ? ' + pr.goal.txt() if pr.goal else ''

def find_essential_cons(g, setup, definitions, translate_to_upper=False):
    essential_points = []
    seen_points = set()
    essential_prems = set()
    essential_cons = set()
    edge = {}
    cons_of_point = {}
    degree = {}
    used_cons = set()
    statement = ""
    for prems, [points] in setup:
        for point in points:
            if point.name not in seen_points:
                essential_points.append(point.name)
                seen_points.add(point.name)
                edge[point.name] = set()
                degree[point.name] = 0
        for prem in prems:
            prem_str = ' '.join(pr.hashed(prem.name, prem.args))
            if prem_str not in essential_prems:
                essential_prems.add(prem_str)

    # print(f"essential points: {essential_points}")
    # print(f"essential prems: {essential_prems}")
    # import pdb; pdb.set_trace()

    while len(essential_points) > 0:
        point = essential_points.pop()
        point = g._name2point[point]
        cons_of_point[point.name] = set()
        flag = False

        for basic in point.basics:
            prem_str = ' '.join(pr.hashed(basic[0], basic[1]))
            con = basic[2].construction
            if con.name == 'midp':
                con.name = 'midpoint'
            cdef = definitions[con.name]
            mapping = dict(zip(cdef.construction.args, con.args))
            new_points = [mapping[point] for point in cdef.points]
            con_str = ' '.join(new_points) + ' = ' + con.txt()
            
            # print(f"point: {point.name}, basic: {prem_str}, con: {con_str}")
            # import pdb; pdb.set_trace()

            # use the current basic to check whether the construction is essential
            if con_str not in essential_cons:
                # if this basic is in the essential prems, then this construction is essential
                if prem_str in essential_prems:
                    essential_cons.add(con_str)
                    # if this construction is essential, then all the points in the args of this construction are essential
                    for arg in con.args:
                        if arg not in seen_points:
                            essential_points.append(arg)
                            seen_points.add(arg)
                            edge[arg] = set()
                            degree[arg] = 0

                    # all the deps of this construction are essential (prems)
                    for dep in cdef.deps.constructions:
                        dep_str = ' '.join([dep.name] + [mapping[arg] for arg in dep.args])
                        essential_prems.add(dep_str)                    

            # if this construction is essential, then use it to construct the point
            if con_str in essential_cons:
                flag = True
                cons_of_point[point.name].add(con_str)
                for arg in con.args:
                    # add an edge from the arg to the point (meaning this point depends on the arg)
                    if point.name not in edge[arg] and arg not in new_points:
                        edge[arg].add(point.name)
                        degree[point.name] += 1

                        # print(f"edge: {arg} -> {point.name}")

        
        assert (flag == True and len(cons_of_point[point.name]) > 0) \
            or (flag == False and len(cons_of_point[point.name]) == 0)
        
        # if none of the constructions are essential, then construct the point using 'free'
        if not flag:
            cons_of_point[point.name].add(f"{point.name} = free {point.name}")
    
    queue = []
    for point in seen_points:
        # print(f"degree {point} = {degree[point]}")
        if degree[point] == 0:
            queue.append(point)
    
    assert len(queue) > 0

    while len(queue) > 0:
        point = queue.pop(0)
        # for con in cons_of_point[point]:
        #     if con not in used_cons:
        #         used_cons.add(con)
        #         statement += f"{con}; "
        con_list = list(cons_of_point[point])
        assert 1 <= len(con_list) <= 2
        con = con_list[0]
        if con not in used_cons:
            used_cons.add(con)
            statement += f"{pretty(con, to_upper=translate_to_upper)}"
            if len(con_list) > 1:
                con = con_list[1]
                statement += ", "
                used_cons.add(con)
                statement += f"{pretty(con, delete_point=True, to_upper=translate_to_upper)}"
            statement += "; "

        for p in edge[point]:
            degree[p] -= 1
            if degree[p] == 0:
                queue.append(p)
    
    return statement[:-2] # remove "; " at the end

def main(argv):

    # definitions of terms used in our domain-specific language.
    ag.DEFINITIONS = pr.Definition.from_txt_file(SHAVE_DEFS_FILE.value, to_dict=True)
    # load inference rules used in DD.
    ag.RULES = pr.Theorem.from_txt_file(SHAVE_RULES_FILE.value, to_dict=True)

    print(len(ag.RULES))

    # load problems from the problems_file,
    problems = pr.Problem.from_txt_file(
        SHAVE_PROBLEMS_FILE.value, to_dict=True, translate=False
    )

    if SHAVE_PROBLEM_NAME.value not in problems:
        raise ValueError(
            f'Problem name `{SHAVE_PROBLEM_NAME.value}` '
            + f'not found in `{SHAVE_PROBLEMS_FILE.value}`'
        )

    this_problem = problems[SHAVE_PROBLEM_NAME.value]

    g, _ = gh.Graph.build_problem(this_problem, ag.DEFINITIONS)
    ag.run_ddar(g, this_problem, SHAVE_OUT_FILE.value)
    setup, _, _, _ = ddar.get_proof_steps(
        g, this_problem.goal, merge_trivials=False
    )
    shaved_statement = find_essential_cons(g, setup, ag.DEFINITIONS)

    print(
        f"\nShaved statement:\n {shaved_statement}"
    )

    shaved_problem_str = this_problem.url + '\n' + shaved_statement + " ? " + this_problem.goal.txt()

    print(
        f"\nShaved problem: \n{shaved_problem_str}"
    )
    print()

    shaved_problem = pr.Problem.from_txt(shaved_problem_str)

    g, _ = gh.Graph.build_problem(shaved_problem, ag.DEFINITIONS)
    ag.run_ddar(g, shaved_problem, SHAVE_OUT_FILE.value)

if __name__ == '__main__':
    # app.run(main)
    definitions = pr.Definition.from_txt_file('../defs.txt', to_dict=True)
    rules = pr.Theorem.from_txt_file('../rules.txt', to_dict=True)
    text = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'
    text = 'A B C D = trapezoid A B C D; E = on_tline E C B A, eqdistance E A C D; F G = trisegment F G A D; H I J = triangle H I J; K = on_line K B H, angle_bisector K A J I; L = on_bline L J D; M = on_bline M C B, eqangle3 M A F G K E; N = on_circle N G F, on_circle N E F; O = intersection_cc O A F K; P = on_pline P I K N, eqdistance P H F M; Q R = square F G Q R; S T U = r_triangle S T U; V = on_line V P G, on_bline V N O ? eqangle A D S U G Q S T'
    text = 'A B C D = trapezoid A B C D; E = on_tline E C B A, eqdistance E A C D; F G = trisegment F G A D; H I J = triangle H I J; K = on_line K B H, angle_bisector K A J I; L = on_bline L J D; M = on_bline M C B, eqangle3 M A F G K E; N = on_circle N G F, on_circle N E F; O = intersection_cc O A F K; P = on_pline P I K N, eqdistance P H F M; Q R = square F G Q R; S T U = r_triangle S T U; V = on_line V P G, on_bline V N O ? eqangle A D S U G Q S T'
    print('[input] ', text)
    shaved_problem = pr.Problem.from_txt(text)
    g, _ = gh.Graph.build_problem(shaved_problem, definitions)
    ddar.solve(g, rules, shaved_problem, max_level=10)

    setup, aux, nl_solution, fl_premises, fl_goal, fl_auxiliary, fl_proof = ag.get_structured_solution(
                        g, 
                        shaved_problem, 
                    )
    shaved_statement = find_essential_cons(g, setup + aux, definitions)
    print('[old shaved statement - main] ', shaved_statement)
    print('[old shaved statement - aux] ', fl_auxiliary.replace('\n', ';'))

    clauses, aux_clauses = ddar.get_essential_clauses(g, shaved_problem.goal)
    print('[new essential clauses - main] ', clauses)
    print('[new essential clauses - aux] ', aux_clauses)

    print('[output] ', find_essential_clauses(g, shaved_problem))