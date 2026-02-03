#!/bin/bash

python tasks/generate_apr.py --output-dir ../output --nsample 5 --num-proc 8
python tasks/evaluate_apr.py --task apr --output-dir ../output
python -m utils.entropy --task program_repair --output-dir ../output
python -m utils.lm_cc_correlation --task program_repair --output-dir ../output