# !/bin/bash
set -e
set -x

# virtualenv -p python3 .
cd .. && source ./bin/activate && cd random_generation

DDAR_ARGS=(
  --shave_defs_file=$(pwd)/../defs.txt \
  --shave_rules_file=$(pwd)/../rules.txt \
);

PROBLEM_FILE="../my_problem.txt"
PROBLEM_NAME="5"

python shave_cons.py --shave_problems_file=$(pwd)/$PROBLEM_FILE --shave_problem_name=$PROBLEM_NAME --shave_defs_file=$(pwd)/../defs.txt --shave_rules_file=$(pwd)/../rules.txt
# --out_file=output/ddar/$PROBLEM_NAME.txt \
# "${DDAR_ARGS[@]}"
