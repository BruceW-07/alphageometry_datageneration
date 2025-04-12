#!/bin/bash

# start_time=$(date +%s)
python generate.py --search_depth=10 --n_threads=10 --samples_per_thread=2 --log_level=info
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log
