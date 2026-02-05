#!/bin/bash

python  -m utils.metrics_correlation --output-dir ../results --task program_repair 
python  -m utils.metrics_correlation --output-dir ../results --task code_translation 
python  -m utils.metrics_correlation --output-dir ../results --task execution_reasoning 