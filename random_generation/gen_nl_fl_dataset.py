import multiprocessing
import os
import sys
sys.path.append('..')
import argparse
import random
import csv

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
from alphageometry import write_solution
from generate_random_proofs import convert_var_names_from_alpha_geo_names
from parse_constrains.get_rand_constrain import ConstraintGenerator


def construct_problem_and_graph(fl_statement, definitions, set_timeout=True):
    try:
        problem = pr.Problem.from_txt(fl_statement)
    except KeyError as e:
        return None, None

    try:
        # Set an alarm for 10 seconds
        if set_timeout:
            signal.alarm(10)

        # Code block to execute with timeout
        graph, _ = gh.Graph.build_problem(problem, definitions)

        # Disable the alarm
        if set_timeout:
            signal.alarm(0)
    except TimeoutException as e:
        print("Graph couldn't be created in reasonable time. Perhaps problem with the premises. Continuing ...")
        # print(f"{fl_statement}")
        return None, None
    except KeyError:
        print("Key error while building graph. Continuing ...")
        print(f"{fl_statement}")
        return None, None
    except ValueError:
        print("Value error while building graph. Continuing ...")
        print(f"{fl_statement}")
        return None, None
    except TypeError:
        print("Some in-compatible goal statement used. Will try another.")
        print(f"{fl_statement}")
        return None, None
    except AttributeError as e:
        print(e)
        print(f"{fl_statement}")
        # TODO(Partha, Max, Felix): This is a hack to avoid the AttributeError. We should fix this.
        return None, None
    except gh.DepCheckFailError:
        print("Dependence check fail while building graph. Continuing ...")
        return None, None

    return problem, graph

def main(run_id, verbose, num_sol_depth):
    dataset_length = 10
    # dataset_length = 100000000
    # filename = f'../../datasets/nl_fl_dataset_{run_id}.csv'
    # filename = (f'/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/geometry_w_proof_mcq_depth{num_sol_depth}/'
    #             f'nl_fl_w_proof_dataset_{run_id}.csv')
    filename = (f'./dataset/geometry_w_proof_mcq_depth{num_sol_depth}/'
                f'nl_fl_w_proof_dataset_{run_id}.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # filename = '../data/nl_fl_dataset_2.csv'
    random.seed(run_id)
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    # field_names = ['sl_n', 'num_clauses', 'nl_statement', 'fl_statement', 'goal_nl', 'goal_fl', 'rnd_states']
    field_names = [
        'sl_n', 
        'num_clauses', 
        'nl_statement', 'fl_statement', 
        'nl_solution', 'fl_solution',
        ]

    # Write data to the CSV file
    with (open(filename, 'w', newline='', encoding='utf-8') as csvfile):
        # writer = csv.DictWriter(csvfile, fieldnames=field_names,)# delimiter='#')
        # this is necessary for inspect to work
        writer = csv.DictWriter(csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"')
        writer.writeheader()
        serial_num = run_id * dataset_length
        cc_gen = CompoundClauseGen(
            definitions, 
            max_comma_sep_clause=2, # setting max_comma_sep_clause > 3 is meaningless
            max_single_clause=7, 
            max_sets=2, 
            seed=run_id,    
            shuffle_var_names=False)
        verbalizer = IndependentStatementVerbalization(None)

        # for i in range(dataset_length):
        while serial_num < (run_id + 1) * dataset_length:
            fl_statement = cc_gen.generate_clauses()
            num_clauses = 0
            for clause in fl_statement.split(';'):
                num_clauses += len(clause.split(','))
            if num_clauses < 5:
                continue

            if verbose: print(fl_statement)

            problem, graph = construct_problem_and_graph(fl_statement, definitions)
            if problem is None or graph is None:
                continue

            if verbose: print(f'Solving ...')

            try:
                ddar.solve(graph, rules, problem, max_level=num_sol_depth)
            except ValueError:
                print("Encountered ValueError while solving. Continuing ...")
                continue
            except (nm.InvalidLineIntersectError, nm.InvalidQuadSolveError):
                print("Encountered InvalidLineIntersectError or InvalidQuadSolveError while solving. Continuing ...")
                continue


            # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
            possible_goals = list(graph.cache.keys())
            if len(possible_goals) > 0:
                goal_fl = list(random.choice(possible_goals)) 

                # get proof
                fl_solution, nl_solution= write_solution(
                    graph, 
                    problem, 
                    goal=pr.Construction(goal_fl[0], list(goal_fl[1:])), 
                    out_file=''
                    )

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
                    'sl_n': serial_num,
                    'num_clauses': num_clauses,
                    'nl_statement': nl_prob + goal_nl,
                    'fl_statement': fl_statement + goal_fl,
                    'nl_solution': nl_solution,
                    'fl_solution': fl_solution,
                })
                serial_num += 1

                print(f'Written {serial_num} rows to {filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create problem fl - nl dataset')
    parser.add_argument('--run_id', required=True, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num_sol_depth', required=True, type=int,
                        help='How many steps will the DDAR search through.')
    parser.add_argument('--n_threads', required=False, type=int, default=1)
    args = parser.parse_args()
    
    if args.verbose:
        main(args.run_id, args.verbose, args.num_sol_depth)
    else:
        with multiprocessing.Pool(args.n_threads) as pool:
            pool.starmap(main, [(args.run_id * args.n_threads + i, args.verbose, args.num_sol_depth)
                                for i in range(args.n_threads)])