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


def merge_datafiles(dir, search_depth):
    csv_files = glob.glob(os.path.join(dir, f'geometry_depth{search_depth}_[0-9]*_.csv'))
    csv_files.sort()
    output_file = os.path.join(dir, f'geometry_depth{search_depth}_raw.csv')
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = None
        idx = 0
        for i, file in enumerate(csv_files):
            with open(file, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                header = next(reader)  # 读取表头
                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header)  # 只写一次表头
                for row in reader:
                    row[0] = idx
                    idx += 1
                    writer.writerow(row)
    output_file2 = os.path.join(dir, f'geometry_depth{search_depth}_pr.txt')
    with open(output_file2, 'w', newline='', encoding='utf-8') as out_f:
        writer = None
        idx = 0
        for i, file in enumerate(csv_files):
            with open(file, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                header = next(reader)  # 读取表头
                for row in reader:
                    row[0] = idx
                    idx += 1
                    out_f.write(str(idx) + '\n')
                    out_f.write(row[3] + '\n')
    output_file3 = os.path.join(dir, f'geometry_depth{search_depth}_llm.txt')
    with open(output_file3, 'w', newline='', encoding='utf-8') as out_f:
        writer = None
        idx = 0
        for i, file in enumerate(csv_files):
            with open(file, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                header = next(reader)  # 读取表头
                for row in reader:
                    idx += 1
                    out_f.write(row[-1] + '\n')
            os.remove(file)
    return idx   

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
                    args = [mapping[a] for a in b.args]
                    name = b.name
                    if b.name in ['s_angle', 'aconst']:
                        x, y, z, v = args
                        name = 'aconst'
                        v = int(v)

                        if v < 0:
                            v = -v
                            x, z = z, x

                        m, n = simplify(int(v), 180)
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
    return data


def run(pid, max_clauses, search_depth, samples_per_thread, dir):
    random.seed(pid)

    # Load definitions and rules
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    # Write data to the CSV file
    filename = os.path.join(dir, f'geometry_depth{search_depth}_{pid}_.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with (open(filename, 'w', newline='', encoding='utf-8') as csvfile):
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

        cc_gen = CompoundClauseGen(
            definitions, 
            max_comma_sep_clause=2, # setting max_comma_sep_clause > 3 is meaningless
            max_single_clause=1, 
            max_sets=max_clauses, 
            seed=pid,    
            shuffle_var_names=False)
        verbalizer = IndependentStatementVerbalization(None)

        idx = 0
        while idx < samples_per_thread:
            # Generate a random problem
            fl_statement = cc_gen.generate_clauses()

            # Find goals
            problem = construct_problem(fl_statement)
            if problem is None: continue
            graph = construct_graph(problem, definitions)
            if graph is None: continue
            try:
                ddar.solve(graph, rules, problem, max_level=search_depth)
            except ValueError:
                logger.debug("Encountered ValueError while solving.")
                continue
            except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
                logger.debug("Encountered InvalidLineIntersectError or InvalidQuadSolveError while solving.")
                continue
            all_goals = list(graph.cache.keys())[::-1]

            # Randomly select a goal
            # goal = list(random.choice(possible_goals))
            # Or ... Find the solution for all goals
            for goal in all_goals:
                if is_naive_goal(goal):
                    continue
                if goal[0] == 'aconst' or goal[0] == 'rconst':
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
                try:
                    ddar.solve(shaved_graph, rules, shaved_problem, max_level=search_depth)
                except ValueError:
                    logger.debug("Encountered ValueError while solving.")
                    continue
                except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
                    logger.debug("Encountered InvalidLineIntersectError or InvalidQuadSolveError while solving.")
                    continue
                try:
                    fl_solution, nl_solution = write_solution(shaved_graph, shaved_problem, '')
                except:
                    logger.warning("Encountered error while writing solution. why???")
                    continue
                if len(fl_solution.split('\n')) < 6:
                    logger.debug("Naive proof") 
                    continue

                # Output problem, goal and proof
                fl_statement_origin = to_upper(fl_statement + ' ? ' + goal.txt())
                fl_problem = to_upper(shaved_problem.txt())
                nl_problem = verbalizer.problem_fl_2_nl(fl_problem) # will ignore goal
                shaved_goal = shaved_problem.goal.txt().split(' ')
                shaved_goal[1:] = [point_name.upper() for point_name in shaved_goal[1:]]
                pretty_goal = pretty_nl(shaved_goal[0], shaved_goal[1:])
                nl_goal = ' Prove that ' + translate_step(pretty_goal)
                try:
                    data = llm_data(shaved_graph, shaved_problem, definitions)
                except:
                    logger.debug("Encountered error while generating llm data. why ???")
                    continue

                writer.writerow({
                    'id': idx,
                    'n_clauses': n_clauses,
                    'fl_statement_full': fl_statement_origin.strip('"'),
                    'fl_statement': fl_problem,
                    'nl_statement': (nl_problem + nl_goal).strip('"'),
                    'nl_solution': nl_solution.strip('"'),
                    'data': llm_data(shaved_graph, shaved_problem, definitions)
                    # shaved_problem.setup_str_from_problem(definitions)+ '\n' +
                })
                logger.info(f'Thread {pid} written sample {idx} to {filename}')
                idx += 1
                if idx == samples_per_thread:
                    break
              

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create problem fl - nl dataset')
    parser.add_argument('--max_clauses', required=True, type=int, default=5)
    parser.add_argument('--search_depth', required=True, type=int,
                        help='How many steps will the DDAR search through.')
    parser.add_argument('--n_threads', required=False, type=int, default=1)
    parser.add_argument('--samples_per_thread', required=False, type=int, default=5)
    parser.add_argument('--dir', default='dataset')
    parser.add_argument('--log_level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()
    
    with open('../common/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    logger.setLevel(getattr(logging, args.log_level.upper()))

    start = time.time()
    if args.n_threads == 1:
        run(0, args.max_clauses, args.search_depth, args.samples_per_thread, args.dir)
    else:
        with multiprocessing.Pool(args.n_threads) as pool:
            pool.starmap(run, [(i, args.max_clauses, args.search_depth, args.samples_per_thread, args.dir) for i in range(args.n_threads)])
    end = time.time()

    n_problems = merge_datafiles(args.dir, args.search_depth)

    logger.info(f'Generate {n_problems} samples in {end - start} seconds.')