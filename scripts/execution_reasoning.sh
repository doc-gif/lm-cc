python tasks/ier/ier.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 1 --writePath ../output/humaneval-ier/initial-output
python tasks/ier/ier.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 2 --writePath ../output/humaneval-ier/initial-output
python tasks/ier/ier.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 3 --writePath ../output/humaneval-ier/initial-output
python tasks/ier/ier.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 4 --writePath ../output/humaneval-ier/initial-output
python tasks/ier/ier.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 5 --writePath ../output/humaneval-ier/initial-output

python tasks/ier/parse_result.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 1
python tasks/ier/parse_result.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 2
python tasks/ier/parse_result.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 3
python tasks/ier/parse_result.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 4
python tasks/ier/parse_result.py --model deepseek-v3 --dataset humaneval --pl Python --task-idx 5

python tasks/ier/get_scores.py --k 1 --output-dir ../output/humaneval-ier --dataset humaneval

python -m utils.entropy --task execution_reasoning --output-dir ../output
python -m utils.lm_cc_correlation --task execution_reasoning --output-dir ../output 
