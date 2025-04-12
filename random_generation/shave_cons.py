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

def find_essential_cons(g, setup, definitions):
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
            prem_str = ' '.join([prem.name] + [p.name for p in prem.args])
            if prem_str not in essential_prems:
                essential_prems.add(prem_str)
    
    while len(essential_points) > 0:
        point = essential_points.pop()
        point = g._name2point[point]
        cons_of_point[point.name] = set()
        flag = False

        for basic in point.basics:
            prem_str = ' '.join([basic[0]] + [p.name for p in basic[1]])
            con = basic[2].construction
            cdef = definitions[con.name]
            mapping = dict(zip(cdef.construction.args, con.args))
            new_points = [mapping[point] for point in cdef.points]
            con_str = ' '.join(new_points) + ' = ' + con.txt()

            # print(f"point: {point.name}, basic: {prem_str}, con: {con_str}")

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
        for con in cons_of_point[point]:
            if con not in used_cons:
                used_cons.add(con)
                statement += f"{con}; "

        for p in edge[point]:
            degree[p] -= 1
            if degree[p] == 0:
                queue.append(p)
    
    return statement[:-2]

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
    app.run(main)