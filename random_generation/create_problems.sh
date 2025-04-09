#!/bin/bash

start_time=$(date +%s)
python gen_nl_fl_dataset.py --run_id $1 --verbose=False --num_sol_depth=5
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
echo "Execution time: $execution_time seconds" >> execution_time.log
