#!/bin/bash

python generate_apr.py --output-dir ../output --nsample 5 --num-proc 8
python evaluate_apr.py --task apr --output-dir ../output
python entropy.py --task program_repair --output-dir ../output