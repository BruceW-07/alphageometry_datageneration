import multiprocessing
import os
import sys
sys.path.append('..')
import random
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

import csv

def is_valid_goal(fl_statement, goal_fl, rules, definitions, set_timeout=True):
    if fl_statement.find('?') < 0:
        premise = fl_statement
    else:
        premise, _ = fl_statement.split('?')

    # try:
    wrong_formal_prob_w_goal = premise.strip() + ' ? ' + goal_fl
    # wrong_formal_prob_w_goal = 'a b c d = r_trapezoid a b c d ? cyclic a b c d'
    problem, graph = construct_problem_and_graph(wrong_formal_prob_w_goal, definitions)
    # except:
    #     try:
    #         import time
    #         time.sleep(5)
    #     except TimeoutException:
    #         import ipdb; ipdb.set_trace()
    #         problem, graph = construct_problem_and_graph(wrong_formal_prob_w_goal, definitions, set_timeout=False)
    if problem is None or graph is None:
        return False
    else:
        try:
            if set_timeout:
                signal.alarm(20)
            g, level_times, status, branches, all_added = ddar.solve(graph, rules, problem, max_level=8)
            # Disable the alarm
            if set_timeout:
                signal.alarm(0)
        except TimeoutException as e:
            # Returning invalid solution if it could not be solved in 30 sec. so the goal is accepted as wrong goals
            return False
    return status == 'solved'


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

def main(run_id, interactive, num_sol_depth):
    # dataset_length = 200
    dataset_length = 100000000
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
    field_names = ['sl_n', 'num_clauses', 'nl_statement', 'fl_statement', 'nl_solution', 'fl_solution','w_goal_nl_1',
                    'w_goal_fl_1', 'w_goal_nl_2', 'w_goal_fl_2', 'w_goal_nl_3', 'w_goal_fl_3']

    # Write data to the CSV file
    wrong_goal_generator = ConstraintGenerator(rules_path)
    with (open(filename, 'w', newline='', encoding='utf-8') as csvfile):
        # writer = csv.DictWriter(csvfile, fieldnames=field_names,)# delimiter='#')
        # this is necessary for inspect to work
        writer = csv.DictWriter(csvfile, fieldnames=field_names, quoting=csv.QUOTE_MINIMAL, quotechar='"')
        writer.writeheader()
        serial_num = run_id * dataset_length
        cc_gen = CompoundClauseGen(definitions, max_comma_sep_clause=2, max_single_clause=7, max_sets=2, seed=run_id,    # setting max_comma_sep_clause > 3 is meaningless
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

            if interactive: print(fl_statement)

            problem, graph = construct_problem_and_graph(fl_statement, definitions)
            if problem is None or graph is None:
                continue

            if interactive: print(f'Solving ...')

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
                goal_fl = list(random.choice(possible_goals))  # comment this line
                # goal_fl = random.choice(possible_goals + [''])  # uncomment this line to get goal less problems
                if goal_fl == '':
                    goal_nl = ''
                    continue
                else:
                    # get proof
                    goal = pr.Construction(goal_fl[0], list(goal_fl[1:]))
                    fl_soln, nl_soln= write_solution(graph, problem, goal=goal, out_file='')

                    capitalized_pt_names = [point_name.capitalize() for point_name in goal_fl[1:]]
                    goal_fl[1:] = capitalized_pt_names
                    # var_map = cc_gen.get_varname_2_alpha_geo_var_map()
                    #only needed when variable names are scrambled. But then should be passed into the proof too
                    # goal_fl = convert_var_names_from_alpha_geo_names(var_map, goal_fl)
                    pretty_goal = pretty_nl(goal_fl[0], goal_fl[1:])
                    if pretty_goal is None:
                        raise ValueError(f'Could not pretty print goal: {goal_fl}')
                    goal_nl = ' Prove that ' + translate_step(pretty_goal)
                    goal_fl = ' ? ' + ' '.join(goal_fl)

                    # generate wrong goals
                    # Initialize lists to store wrong goals
                    w_goals_nl = ['None', 'None', 'None']
                    w_goals_fl = ['None', 'None', 'None']

                    # Generate 3 different wrong goals
                    valid_goal_count = 0
                    for _ in range(10):
                        w_goal_nl_temp, w_goal_fl_temp = \
                        get_wrong_goal_nl_fl(capitalized_pt_names, wrong_goal_generator)
                        # we are looking for wrong goals!
                        if interactive: print(f'Validating \n {fl_statement}\nwith goal\n{w_goal_fl_temp}')
                        if not is_valid_goal(fl_statement, w_goal_fl_temp, rules, definitions):
                            w_goals_nl[valid_goal_count] = w_goal_nl_temp
                            w_goals_fl[valid_goal_count] = w_goal_fl_temp
                            valid_goal_count += 1
                            if valid_goal_count == 3:
                                break

                # Now we know that the generated premises are not contradictory
                # nl_prob = get_nl_problem_statement(fl_statement)
                nl_prob = verbalizer.problem_fl_2_nl(fl_statement)
                # dump this row
                row = {
                    'sl_n': serial_num,
                    'num_clauses': num_clauses,
                    'nl_statement': nl_prob + goal_nl,
                    'fl_statement': fl_statement + goal_fl,
                    'nl_solution': nl_soln,
                    'fl_solution': fl_soln,
                    'w_goal_nl_1': w_goals_nl[0],
                    'w_goal_fl_1': w_goals_fl[0],
                    'w_goal_nl_2': w_goals_nl[1],
                    'w_goal_fl_2': w_goals_fl[1],
                    'w_goal_nl_3': w_goals_nl[2],
                    'w_goal_fl_3': w_goals_fl[2],
                }
                writer.writerow(row)
                serial_num += 1


def get_wrong_goal_nl_fl(capitalized_pt_names, wrong_goal_generator):
    wrong_goal = wrong_goal_generator.generate_constraint(capitalized_pt_names)
    wrong_goals_list = wrong_goal.split(' ')
    pretty_wrong_goal = pretty_nl(wrong_goals_list[0], wrong_goals_list[1:])
    if pretty_wrong_goal is None:
        raise ValueError(f'Could not pretty print wrong goal: {wrong_goal}')
    wrong_goal_nl = ' Prove that ' + translate_step(pretty_wrong_goal)

    return wrong_goal_nl, wrong_goal


def str_to_bool(value):
    if value.lower() in ['true', 't', 'yes', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', '0', 'flase']:  # Including common typo
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create problem fl - nl dataset')
    parser.add_argument('--run_id', required=True, type=int, help='An integer positional argument')
    parser.add_argument('--interactive', required=True, type=str_to_bool,
                        help='A boolean value (true/false)')
    parser.add_argument('--num_sol_depth', required=True, type=int,
                        help='Howmany steps will the DDAR search through.')
    args = parser.parse_args()

    n_processes = 16
    offset = 0 * n_processes

    if args.interactive:
        main(args.run_id, args.interactive, args.num_sol_depth)
    else:
        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(main, [(offset + args.run_id * n_processes + i, args.interactive, args.num_sol_depth)
                                for i in range(n_processes)])

