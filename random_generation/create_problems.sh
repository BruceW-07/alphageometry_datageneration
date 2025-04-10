#!/bin/bash

start_time=$(date +%s)
python gen_nl_fl_dataset.py --run_id $1 --num_sol_depth=5 --n_threads=1 # --verbose
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
echo "Execution time: $execution_time seconds" >> execution_time.log
