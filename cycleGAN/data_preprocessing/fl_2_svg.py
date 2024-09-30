import sys
sys.path.append('../../')
from utils.loading_utils import load_definitions_and_rules
import graph as gh
import problem as pr
from random_generation.draw_svg.get_svg import draw_svg
from cycleGAN.my_utils.point_naming_util import rename_point_names, reverse_mapping
import pandas as pd
import os
import tqdm
from multiprocessing import Pool
import signal

class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)

def remove_empty_lines(text):
    """
    Removes all empty lines from the given text.
    """
    return "\n".join([line for line in text.splitlines() if line.strip()])

def process_row(args):
    idx, fl_statement = args
    # Load definitions inside the worker
    defs_path = '../../defs.txt'
    rules_path = '../../rules.txt'
    definitions, _ = load_definitions_and_rules(defs_path, rules_path)

    org_2_alpha_geo, renamed_txt = rename_point_names(set(definitions.keys()), fl_statement)
    alpha_geo_2_org = reverse_mapping(org_2_alpha_geo)
    prob_prem, goal_str = fl_statement.split('?')
    goal_str = ''
    svg_text = draw_problem(alpha_geo_2_org, definitions, goal_str, org_2_alpha_geo, prob_prem)
    svg_text = remove_empty_lines(svg_text)
    return idx, svg_text

def draw_problem(alpha_geo_2_org, definitions, goal_str, org_2_alpha_geo, prob_prem):
    svg_text = ''
    try:
        signal.alarm(20)
        problem = pr.Problem.from_txt(prob_prem.strip())
        graph, _ = gh.Graph.build_problem(problem, definitions)
        svg_text = draw_svg(
            graph.type2nodes[gh.Point],
            graph.type2nodes[gh.Line],
            graph.type2nodes[gh.Circle],
            graph.type2nodes[gh.Segment],
            alpha_geo_2_org=alpha_geo_2_org,
            org_2_alpha_geo=org_2_alpha_geo,
            goal_str=goal_str.strip())
        signal.alarm(0)
    except AttributeError:
        svg_text = ''
    except TimeoutException as e:
        print("Graph couldn't be create in reasonable time. Perhaps problem with the premises. Exiting ...")

    return svg_text


def main():
    # Read the CSV file
    dataset_root = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/'
    destination_path = os.path.join(dataset_root, 'svg_fl_long.csv')
    if os.path.exists(destination_path):
        raise ValueError(f'File {destination_path} already exists')
    df = pd.read_csv(os.path.join(dataset_root, 'nl_fl_long.csv'))

    # Prepare iterable for multiprocessing
    iterable = [(idx, fl_statement) for idx, fl_statement in enumerate(df['fl_statement'])]

    # Use multiprocessing with 24 processes
    with Pool(processes=16) as pool:
        # Process rows in parallel with increased chunksize
        results = list(tqdm.tqdm(pool.imap(process_row, iterable, chunksize=1000), total=len(iterable)))

    # Update DataFrame with results
    rows_to_delete = []
    for idx, svg_text in results:
        if svg_text == '':
            rows_to_delete.append(idx)
        else:
            df.at[idx, 'nl_statement'] = svg_text

    # Save the updated DataFrame to a new CSV file
    df = df.drop(rows_to_delete).reset_index(drop=True)
    df.to_csv(destination_path, index=False)

if __name__ == '__main__':
    main()
