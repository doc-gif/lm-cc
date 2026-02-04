#!/bin/bash

python  -m utils.metrics_correlation --output-dir ../ouroutput --task program_repair 
python  -m utils.metrics_correlation --output-dir ../ouroutput --task code_translation 
python  -m utils.metrics_correlation --output-dir ../ouroutput --task execution_reasoning 