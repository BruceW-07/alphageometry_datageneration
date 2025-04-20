#!/bin/bash

MELIAD_PATH=$(pwd)/../meliad_lib/meliad
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

# start_time=$(date +%s)
python generate.py --max_clauses=10 --search_depth=4 --n_threads=1 --n_samples=10 --log_level=info
python analyze.py 
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log
