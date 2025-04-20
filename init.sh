# !/bin/bash
set -e
set -x

# Check if Python version is 3.10
python_version=$(python --version 2>&1 | awk '{print $2}')
major_minor_version=$(echo $python_version | cut -d. -f1,2)

if [ "$major_minor_version" != "3.10" ]; then
  echo "Error: This script requires Python 3.10"
  echo "Current Python version: $python_version"
  exit 1
fi

echo "Python 3.10 detected, proceeding with installation..."

pip install --require-hashes -r requirements.txt

pip install huggingface_hub
huggingface-cli download abrahamabelboodala/ALPHAGEOMETRY_ag_ckpt_vocab --local-dir ALPHAGEOMETRY_ag_ckpt_vocab --local-dir-use-symlinks False
mv ALPHAGEOMETRY_ag_ckpt_vocab/ag_ckpt_vocab .
rm -rf ALPHAGEOMETRY_ag_ckpt_vocab
DATA=ag_ckpt_vocab

MELIAD_PATH=meliad_lib/meliad
mkdir -p $MELIAD_PATH
git clone https://github.com/google-research/meliad $MELIAD_PATH
export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH