import multiprocessing
import os
import sys
sys.path.append('..')
import argparse
import random
import time
import csv
import yaml
import glob
from collections import defaultdict
import logging, logging.config
logger = logging.getLogger("generation")
    
import ddar
import graph as gh
import numericals as nm
import problem as pr
from clause_generation import CompoundClauseGen
import signal
import pretty as pt
from generate_random_proofs import TimeoutException
from utils.loading_utils import load_definitions_and_rules
from prettier_print.pretty_problem_statement import get_nl_problem_statement
from pretty import pretty_nl
from prettier_print.prettier_proof_statements import translate_step
from verb.verbalize import IndependentStatementVerbalization
from alphageometry import write_solution, get_structured_solution


def construct_problem(fl_statement):
    try:
        problem = pr.Problem.from_txt(fl_statement)
    except KeyError as e:
        return None
    return problem

def construct_graph(problem, definitions, timeout=10):
    try:
        # Set alarm
        signal.alarm(timeout)
        # Code block to execute with timeout
        graph, _ = gh.Graph.build_problem(problem, definitions)
        # Disable the alarm
        signal.alarm(0)
    except TimeoutException as e:
        logging.debug("Graph couldn't be created in reasonable time.")
        return None
    except KeyError:
        logging.debug("Key error while building graph. ")
        return None
    except ValueError:
        logging.debug("Value error while building graph. ")
        return None
    except TypeError:
        logging.debug("Some in-compatible goal statement used. Will try another.")
        return None
    except AttributeError as e:
        logging.debug(e)
        # TODO(Partha, Max, Felix): This is a hack to avoid the AttributeError. We should fix this.
        return None
    except gh.DepCheckFailError:
        logger.debug("Dependence check fail while building graph. ")
        return None
    return graph

def solve_problem(graph, rules, problem, search_depth):
    try:
        ddar.solve(graph, rules, problem, max_level=search_depth)
    except ValueError:
        logger.debug("Encountered ValueError while solving.")
        return False
    except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
        logger.debug("Encountered InvalidLineIntersectError or InvalidQuadSolveError while solving.")
        return False
    except:
        logger.debug("Encountered error while solving.")
        return False    
    return True

def is_naive_goal(goal):
    # case1: cong AB = AB, para AB ∥∥ AB, rconst AB:AB=1, aconst ∠AB AB=0
    if goal[0] == 'cong' or goal[0] == 'para' or goal[0] == 'rconst' or goal[0] == 'aconst':
        left = {goal[1], goal[2]}
        right = {goal[3], goal[4]}
        if left == right:
            return True
    elif goal[0] == 'eqratio':
        #case2: eqratio AB/CD = DC/BA, eqangle ∠AB CD = ∠DC/BA
        seg_1 = {goal[1], goal[2]}
        seg_2 = {goal[3], goal[4]}
        seg_3 = {goal[5], goal[6]}
        seg_4 = {goal[7], goal[8]}
        if seg_1 == seg_3 and seg_2 == seg_4:
            return True
        if seg_1 == seg_4 and seg_2 == seg_3:
            return True
    return False

def to_upper(fl_statement):
    statement, goal = fl_statement.split(' ? ')
    clauses = statement.split('; ')
    statement = ''
    for clause in clauses:
        points, cons = clause.split(' = ')
        statement += points.upper() + ' = '
        cons = cons.split(', ')
        for i in range(len(cons)):
            con = cons[i].split(' ')
            if con[0] == 'midp':
                con[0] = 'midpoint'
            statement += con[0] + ' '
            for j in range(1, len(con)):
                statement += con[j].upper();
                if j != len(con) - 1:
                    statement += ' '
            if i != len(cons) - 1:
                statement += ', '
        statement += '; '
    statement = statement[:-2]
    goal = goal.split(' ')
    goal[1:] = [point.upper() for point in goal[1:]]
    return statement + ' ? ' + ' '.join(goal)

def llm_data(graph, problem, definitions):
    """Construct the <theorem_premises> string from Problem object."""
    clauses, aux_clauses, points_list, aux_points_list = ddar.get_essential_clauses(graph, problem.goal)

    string = []
    data_tmp = defaultdict(list)
    points_premise = set()
    for clause in problem.clauses:
        group = {}
        p2deps = defaultdict(list)
        for c in clause.constructions:
            cdef = definitions[c.name]

            if len(c.args) != len(cdef.construction.args):
                assert len(c.args) + len(clause.points) == len(cdef.construction.args)
                c.args = clause.points + c.args

            mapping = dict(zip(cdef.construction.args, c.args))
            for points, bs in cdef.basics:
                points = tuple([mapping[x] for x in points])
                for p in points:
                    group[p] = points

                for b in bs:
                    if b.name == 'rconst' and c.name == 'triangle12':
                        args = [mapping[a] for a in b.args[:-2]]
                        args.append(0.5)
                    else:
                        args = [mapping[a] for a in b.args]
                    name = b.name
                    if b.name in ['s_angle', 'aconst']:
                        x, y, z, v = args
                        name = 'aconst'
                        v = int(v)

                        if v < 0:
                            v = -v
                            x, z = z, x

                        m, n = pr.simplify(int(v), 180)
                        args = [y, z, y, x, f'{m}pi/{n}']

                    p2deps[points].append(pr.hashed_txt(name, args))

        for k, v in p2deps.items():
            p2deps[k] = pr.sort_deps(v)

        points = clause.points
        while points:
            p = points[0]
            gr = group[p]
            points = [x for x in points if x not in gr]

            deps_str = []
            for dep in p2deps[gr]:
                dep_str = pt.pretty(dep)

                if dep[0] == 'aconst':
                    m, n = map(int, dep[-1].split('pi/'))
                    mn = f'{m}. pi / {n}.'
                    dep_str = ' '.join(dep_str.split()[:-1] + [mn])
                deps_str.append(dep_str)
            
            data_tmp[' '.join(gr)] = deps_str
                
    data = '{S} '
    ref = 0
    string_premise = []
    for k, v in data_tmp.items():
        if not all(p in aux_points_list for p in k.split(' ')):
            v = [s + ' {:02}'.format(ref+i) for i, s in enumerate(v)]
            ref += len(v)
            string_premise.append(k + ' : ' + ' '.join(v))
    data += ' ; '.join([s.strip() for s in string_premise])

    data += ' ? '+ pt.pretty([problem.goal.name] + problem.goal.args)

    string_aux = []
    for k, v in data_tmp.items():
        if all(p in aux_points_list for p in k.split(' ')):
            v = [s + ' {:02}'.format(ref+i) for i, s in enumerate(v)]
            ref += len(v)
            string_aux.append(k + ' : ' + ' '.join(v))
    if len(string_aux) > 0:
        data += ' {F1} x00 '
        data += ' ; '.join([s.strip() for s in string_aux])
        data += ' ;'
    return data

def run(args):
    """Process a single problem generation task.
    
    Returns:
        A list of generated data items
    """
    pid, fl_statement, search_depth = args
    random.seed(pid)

    # Load definitions and rules
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    # Create a list to store the generated data
    generated_data = [] 

    # Find goals
    problem = construct_problem(fl_statement)
    if problem is None: return []
    graph = construct_graph(problem, definitions)
    if graph is None: return []
    if not solve_problem(graph, rules, problem, search_depth): return []
    all_goals = list(graph.cache.keys())[::-1]

    # Randomly select a goal
    # goal = list(random.choice(possible_goals))
    # Or ... Find the solution for all goals
    for goal in all_goals:
        if is_naive_goal(goal):
            continue
        if goal[0] == 'aconst' or goal[0] == 'rconst':
            # AlphaGeometry 1 不支持 aconst 和 rconst
            logger.debug("Goal is 'aconst' or 'rconst'. Skip this problem.")
            continue
        goal = pr.Construction(goal[0], list(goal[1:]))
        # Get essential clauses for each goal
        clauses, aux_clauses, _, _ = ddar.get_essential_clauses(graph, goal)
        shaved_statement = []
        for clause in problem.clauses:
            if clause.txt() in clauses or clause.txt() in aux_clauses:
                shaved_statement.append(clause.txt())
        n_clauses = len(shaved_statement)
        shaved_statement = '; '.join(shaved_statement) + ' ? ' + goal.txt() if goal else ''

        # Rename points and get solution
        # import pdb; pdb.set_trace()
        shaved_problem = construct_problem(shaved_statement)
        if shaved_problem is None: continue
        shaved_graph = construct_graph(shaved_problem, definitions)
        if shaved_graph is None: continue
        if not solve_problem(shaved_graph, rules, shaved_problem, search_depth): continue
        try:
            fl_solution, nl_solution = write_solution(shaved_graph, shaved_problem, '')
        except:
            logger.warning("Encountered error while writing solution. why???")
            logger.warning("fl_statement\n", fl_statement)
            logger.warning("shaved_statement\n", shaved_statement)
            continue
        if len(fl_solution.split('\n')) < 6:
            logger.debug("Naive proof") 
            continue

        # Output problem, goal and proof
        verbalizer = IndependentStatementVerbalization(None)
        fl_statement_origin = to_upper(fl_statement + ' ? ' + goal.txt())
        fl_problem = to_upper(shaved_problem.txt())
        nl_problem = verbalizer.problem_fl_2_nl(fl_problem) # will ignore goal
        shaved_goal = shaved_problem.goal.txt().split(' ')
        shaved_goal[1:] = [point_name.upper() for point_name in shaved_goal[1:]]
        pretty_goal = pretty_nl(shaved_goal[0], shaved_goal[1:])
        nl_goal = ' Prove that ' + translate_step(pretty_goal)
        try:
            data = llm_data(shaved_graph, shaved_problem, definitions)
            # data = shaved_problem.setup_str_from_problem(definitions)+ '\n' +
        except:
            logger.warning("Encountered error while generating llm data. why ???")
            # logger.warning("fl_statement\n", fl_statement)
            # logger.warning("shaved_statement\n", shaved_statement)
            continue

        # Store the data instead of writing it directly
        generated_data.append({
            'n_clauses': n_clauses,
            'fl_statement_full': fl_statement_origin.strip('"'),
            'fl_statement': fl_problem,
            'nl_statement': (nl_problem + nl_goal).strip('"'),
            'nl_solution': nl_solution.strip('"'),
            'data': data
        })
    
    return generated_data

def write_data(all_data, dir, search_depth):
    """Write all generated data to output files."""
    filename = os.path.join(dir, f'geometry_depth{search_depth}_raw.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        field_names = [
            'id', 
            'n_clauses', 
            'fl_statement_full',
            'fl_statement', 
            'nl_statement', 
            'nl_solution', 
            'data',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"')
        writer.writeheader()
        
        for i, row in enumerate(all_data):
            row['id'] = i
            writer.writerow(row)
    
    # Write to pr.txt
    with open(os.path.join(dir, f'geometry_depth{search_depth}_pr.txt'), 'w', 
              encoding='utf-8') as out_f:
        for i, row in enumerate(all_data):
            out_f.write(f"{i}\n{row['fl_statement']}\n")
    
    # Write to llm.txt
    with open(os.path.join(dir, f'geometry_depth{search_depth}_llm.txt'), 'w', 
              encoding='utf-8') as out_f:
        for row in all_data:
            out_f.write(row['data'] + '\n')
    
    return len(all_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create problem fl - nl dataset')
    parser.add_argument('--max_clauses', required=True, type=int, default=5)
    parser.add_argument('--search_depth', required=True, type=int,
                        help='How many steps will the DDAR search through.')
    parser.add_argument('--n_threads', required=False, type=int, default=1)
    parser.add_argument('--n_samples', required=False, type=int, default=5)
    parser.add_argument('--dir', default='dataset')
    parser.add_argument('--log_level', default='info', 
                       choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()
    
    with open('../common/logging.yaml', 'r') as f:
        logging.config.dictConfig(yaml.safe_load(f))
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Initialize clause generator
    definitions, rules = load_definitions_and_rules('../defs.txt', '../rules.txt')
    cc_gen = CompoundClauseGen(
        definitions, 
        max_comma_sep_clause=2, # setting max_comma_sep_clause > 3 is meaningless
        max_single_clause=1, 
        max_sets=args.max_clauses,
        seed=0,  
        shuffle_var_names=False)
    
    # Create task generator
    task_generator = ((i, cc_gen.generate_clauses(), args.search_depth) 
                     for i in range(10**9))
    
    # Generate data
    all_data = []
    start = time.time()
    if args.n_threads == 1:
        task_iterator = iter(task_generator)
        while len(all_data) < args.n_samples:
            result = run(next(task_iterator))
            all_data.extend(result)
            if result:
                logger.info(f'Generated {len(all_data)} samples in {time.time() - start:.1f}s '
                            f'({(time.time() - start)/len(all_data):.1f}s/sample)')
    else:
        with multiprocessing.Pool(args.n_threads) as pool:
            for result in pool.imap_unordered(run, task_generator):
                all_data.extend(result)
                if result:
                    logger.info(f'Generated {len(all_data)} samples in {time.time() - start:.1f}s '
                              f'({(time.time() - start)/len(all_data):.1f}s/sample)')
                if len(all_data) >= args.n_samples:
                    pool.terminate()
                    pool.join()
                    break
    
    # Write results
    n_problems = write_data(all_data, args.dir, args.search_depth)
    