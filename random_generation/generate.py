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
import logging, logging.config
logger = logging.getLogger("generation")
    
import ddar
import graph as gh
import numericals as nm
import problem as pr
from clause_generation import CompoundClauseGen
import signal
from generate_random_proofs import TimeoutException
from utils.loading_utils import load_definitions_and_rules
from prettier_print.pretty_problem_statement import get_nl_problem_statement
from pretty import pretty_nl
from prettier_print.prettier_proof_statements import translate_step
from utils.get_rand_gen_states import get_random_states
from verb.verbalize import IndependentStatementVerbalization
from alphageometry import write_solution, get_structured_solution
from generate_random_proofs import convert_var_names_from_alpha_geo_names
from parse_constrains.get_rand_constrain import ConstraintGenerator
from shave_cons import find_essential_cons

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

def merge_datafiles(dir, search_depth):
    csv_files = glob.glob(os.path.join(dir, f'geometry_depth{search_depth}_*.csv'))
    csv_files.sort()
    output_file = os.path.join(dir, f'geometry_depth{search_depth}.csv')
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = None
        for i, file in enumerate(csv_files):
            with open(file, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                header = next(reader)  # 读取表头
                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header)  # 只写一次表头
                for row in reader:
                    writer.writerow(row)



def run(pid, search_depth, samples_per_thread, dir):
    random.seed(pid)

    # Load definitions and rules
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    # Write data to the CSV file
    filename = os.path.join(dir, f'geometry_depth{search_depth}_{pid}.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with (open(filename, 'w', newline='', encoding='utf-8') as csvfile):
        field_names = [
            'id', 
            'n_clauses', 
            'nl_statement', 
            'fl_statement', 
            'nl_solution', 
            'fl_premises',
            'fl_goal',
            'fl_auxiliary',
            'fl_proof'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"')
        writer.writeheader()

        cc_gen = CompoundClauseGen(
            definitions, 
            max_comma_sep_clause=2, # setting max_comma_sep_clause > 3 is meaningless
            max_single_clause=1, 
            max_sets=5, 
            seed=pid,    
            shuffle_var_names=False)
        verbalizer = IndependentStatementVerbalization(None)

        shaved = False
        shaved_statement = ''
        sid = pid * samples_per_thread
        while sid < (pid + 1) * samples_per_thread:
            if shaved:
                fl_statement = shaved_statement
            else:
                fl_statement = cc_gen.generate_clauses()
                
            n_clauses = len(fl_statement.split(';'))
            if not shaved_statement and n_clauses < 4: continue

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

            # Randomly select a cache node to be the goal. 
            # TODO: Is this right can we do better? Consider coverage!
            possible_goals = list(graph.cache.keys())
            if len(possible_goals) > 0:
                goal_fl = list(random.choice(possible_goals))
                if is_naive_goal(goal_fl):
                    logger.debug("Naive goal like AB = BA")
                    continue

                # get proof
                setup, nl_solution, fl_premises, fl_goal, fl_auxiliary, fl_proof = get_structured_solution(
                    graph, 
                    problem, 
                    goal=pr.Construction(goal_fl[0], list(goal_fl[1:])), 
                )
                if fl_premises == '':# or fl_premises == ': Points\n':
                    logger.debug("Naive proof using premises from clauses directly") 
                    continue
                       
                if not shaved:
                    try:
                        signal.alarm(10)
                        shaved_statement = find_essential_cons(graph, setup, definitions)
                        signal.alarm(0)
                    except:
                        logging.debug("Graph couldn't be shaved in reasonable time.")
                        shaved = False
                        continue
                    shaved = True
                    continue # to rename points
                
                goal_fl[1:] = [point_name.capitalize() for point_name in goal_fl[1:]]
                # var_map = cc_gen.get_varname_2_alpha_geo_var_map()
                #only needed when variable names are scrambled. But then should be passed into the proof too
                # goal_fl = convert_var_names_from_alpha_geo_names(var_map, goal_fl)
                pretty_goal = pretty_nl(goal_fl[0], goal_fl[1:])
                goal_nl = ' Prove that ' + translate_step(pretty_goal)
                goal_fl = ' ? ' + ' '.join(goal_fl)
                # nl_prob = get_nl_problem_statement(fl_statement)
                nl_prob = verbalizer.problem_fl_2_nl(fl_statement)

                writer.writerow({
                    'id': sid,
                    'n_clauses': n_clauses,
                    'nl_statement': nl_prob + goal_nl,
                    'fl_statement': fl_statement + goal_fl,
                    'nl_solution': nl_solution,
                    'fl_premises': fl_premises,
                    'fl_goal': fl_goal,
                    'fl_auxiliary': fl_auxiliary,
                    'fl_proof': fl_proof
                })
                logger.info(f'Written sample {sid} to {filename}')
                sid += 1
                shaved = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create problem fl - nl dataset')
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
        run(0, args.search_depth, args.samples_per_thread, args.dir)
    else:
        with multiprocessing.Pool(args.n_threads) as pool:
            pool.starmap(run, [(i, args.search_depth, args.samples_per_thread, args.dir) for i in range(args.n_threads)])
    end = time.time()

    merge_datafiles(args.dir, args.search_depth)

    logger.info(f'Generate {args.n_threads * args.samples_per_thread} samples in {end - start} seconds.')