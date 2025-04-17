#!/bin/bash

# start_time=$(date +%s)
python generate.py --max_clauses=10 --search_depth=5 --n_threads=10 --samples_per_thread=10 --log_level=info
python analyze.py 
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log
