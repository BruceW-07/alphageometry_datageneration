#!/bin/bash

# start_time=$(date +%s)
python generate_dataset.py --run_id 0 --search_depth=5 --n_threads=1 --samples_per_thread=10 --log_level=info
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log
