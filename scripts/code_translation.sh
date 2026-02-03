#!/bin/bash

python tasks/generate_ct.py --output-dir ../output --nsample 5 --num-proc 8
python tasks/evaluate_ct.py --task code_translation --output-dir ../output
python -m utils.entropy --task code_translation --output-dir ../output
python -m utils.lm_cc_correlation --task code_translation --output-dir ../output 