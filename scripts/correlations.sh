#!/bin/bash

python  -m utils.metrics_correlation --output-dir ../ouroutput --task program_repair --cache
python  -m utils.metrics_correlation --output-dir ../ouroutput --task code_translation --cache
python  -m utils.metrics_correlation --output-dir ../ouroutput --task execution_reasoning --cache