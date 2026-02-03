import numpy as np
import argparse
import os
import json
from .correlation import  get_grouped_partial_corr, read_loc, read_lmcc, print_metric_result


def print_metric_result(all_score, all_metric, all_loc, option="both"):
    if option == "zero" or option == "both":
        res_zero, _ = get_grouped_partial_corr(all_score, all_metric,  None)    
        r_zero = None
        p_zero = None
        if res_zero:
            r_zero = round(res_zero["partial_correlation"]["spearman-r"], 2)
            p_zero = round(res_zero["partial_correlation"]["spearman-pval"], 2)
        print(f"Zero: {r_zero}({p_zero})")
    if option == "partial" or option == "both":
        res_partial, _ = get_grouped_partial_corr(all_score, all_metric, all_loc)
        r_partial = None
        p_partial = None
        if res_partial:
            r_partial = round(res_partial["partial_correlation"]["spearman-r"], 2)
            p_partial = round(res_partial["partial_correlation"]["spearman-pval"], 2)
        print(f"Partial: {r_partial}({p_partial})")
    print("\n")



def get_lm_cc(output_dir, dataset, block_tree_file_path, results_file_path, dataset_output_dir):
    id2score = {}
    id2loc = {}
    id2lmcc = {}


    with open(block_tree_file_path, 'r') as f:
        block_trees = json.load(f)
    
    if dataset == "humaneval-ier":
        with open(results_file_path, 'r') as f:
            id2score = json.load(f)
    elif dataset == "xcodeeval-apr":
        with open(results_file_path, 'r') as f:
            results_data = json.load(f)
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            result = results_data[task_id]
            id2score[task_id] = result["pass@1"]
    elif dataset == "xcodeeval-ct":
        with open(results_file_path, 'r') as f:
            results_data = json.load(f)
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            result = results_data[task_id]
            id2score[task_id] = result["pass@1"]
    
    loc_file_path = os.path.join(dataset_output_dir, 'loc.json')
    lmcc_file_path = os.path.join(dataset_output_dir, 'lmcc.json')

    id2loc = read_loc(loc_file_path, block_trees, calculate)
    id2lmcc = read_lmcc(lmcc_file_path, block_trees, calculate)
    
    all_score = np.array([])
    all_loc = np.array([])
    all_lmcc = np.array([])


    for treeinfo in block_trees:
        task_id = f"{treeinfo['task_id']}"

        all_score = np.append(all_score, id2score[task_id])
        all_loc = np.append(all_loc, id2loc[task_id])
        all_lmcc = np.append(all_lmcc, id2lmcc[task_id])

    print("="*15, "LM-CC", "="*15)
    print_metric_result(all_score, all_lmcc, all_loc, option="partial")



def main(task, output_dir, calculate):
    THRESHOLD = 0.67
    model_name = "CodeLlama-7b-hf"
    TEMPERATURE = 1.0
    block_tree_filename = f"block_trees-{model_name}-{TEMPERATURE}-thres-{THRESHOLD}.json"
    if task == "program_repair":
        dataset = "xcodeeval-apr"
        dataset_output_dir = os.path.join(output_dir, "xcodeeval", "apr")
        results_file_path = os.path.join(dataset_output_dir, "python_test_filtered_results.json")
    elif task == "code_translation":
        dataset = "xcodeeval-ct"
        dataset_output_dir = os.path.join(output_dir, "xcodeeval", "code_translation")
        results_file_path = os.path.join(dataset_output_dir, "python2c_test_filtered_results.json")
    elif task == "execution_reasoning":
        dataset = "humaneval-ier"
        dataset_output_dir = os.path.join(output_dir, dataset)
        results_file_path = os.path.join(dataset_output_dir, "results_score.json")

    block_tree_file_path = os.path.join(dataset_output_dir, "entropy", block_tree_filename)

    get_lm_cc(output_dir, dataset, block_tree_file_path, results_file_path, dataset_output_dir)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default='xcodeeval-apr', help="select one from [xcodeeval-apr, xcodeeval-ct, humaneval-ier]")
    parser.add_argument("--task", type=str, default='program_repair', help="select one from [program_repair, code_translation, execution_reasoning]")
    parser.add_argument("--output-dir", type=str, default="../output")
    parser.add_argument("--cache", action='store_true')
    args = parser.parse_args()

    task = args.task
    output_dir = args.output_dir

    calculate = not args.cache
    main(task, output_dir, calculate)