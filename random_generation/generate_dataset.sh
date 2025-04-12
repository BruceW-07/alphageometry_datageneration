#!/bin/bash

# start_time=$(date +%s)
python generate_dataset.py --search_depth=5 --n_threads=2 --samples_per_thread=3 --log_level=info
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds"
# echo "Execution time: $execution_time seconds" >> execution_time.log
