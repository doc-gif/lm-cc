import numpy as np
import argparse
import os
import json
import sys
import os

from .utils import get_depth_sum, get_total_branch, get_max_depth, get_max_width, get_avg_children, get_avg_depth
from .correlation import  read_loc, read_cc, read_mi, read_hc_difficulty, read_cog, read_lmcc, print_metric_result





def get_metrics(task, dataset, block_tree_file_path, results_file_path, dataset_output_dir, calculate):
    id2score = {}
    id2cc = {}
    id2loc = {}
    id2cog = {}
    id2mi = {}
    id2hc_difficulty = {}
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
    
    cc_file_path = os.path.join(dataset_output_dir, 'cc.json')
    loc_file_path = os.path.join(dataset_output_dir, 'loc.json')
    cog_file_path = os.path.join(dataset_output_dir, 'cog.json')
    mi_file_path = os.path.join(dataset_output_dir, 'mi.json')
    hc_difficulty_file_path = os.path.join(dataset_output_dir, 'hc_difficulty.json')
    lmcc_file_path = os.path.join(dataset_output_dir, 'lmcc.json')

    id2cc = read_cc(cc_file_path, block_trees, calculate)
    id2loc = read_loc(loc_file_path, block_trees, calculate)
    id2cog = read_cog(cog_file_path, block_trees, calculate)
    id2mi = read_mi(mi_file_path, block_trees, calculate)
    id2hc_difficulty = read_hc_difficulty(hc_difficulty_file_path, block_trees, calculate)
    id2lmcc = read_lmcc(lmcc_file_path, block_trees, calculate)


    
    all_score = np.array([])
    all_loc = np.array([])
    all_cc = np.array([])
    all_lmcc = np.array([])
    all_cog = np.array([])
    all_mi = np.array([])
    all_hc_difficulty = np.array([])

    all_max_comp_leval = np.array([])   # max depth
    all_avg_comp_leval = np.array([])   # avg  depth
    all_total_comp_leval = np.array([]) # depth sum
    
    all_max_branch  = np.array([]) # max width
    all_avg_branch  = np.array([]) # avg child cnt
    all_total_branch  = np.array([]) # block cnt-1

    for treeinfo in block_trees:
        task_id = f"{treeinfo['task_id']}"
        block_tree = treeinfo["block_tree"]

        # all_depths = np.append(all_depths, depth)
        all_score = np.append(all_score, id2score[task_id])
        all_loc = np.append(all_loc, id2loc[task_id])
        all_cc = np.append(all_cc, id2cc[task_id])
        all_lmcc = np.append(all_lmcc, id2lmcc[task_id])
        all_cog = np.append(all_cog, id2cog[task_id])
        all_mi = np.append(all_mi, id2mi[task_id])
        all_hc_difficulty = np.append(all_hc_difficulty, id2hc_difficulty[task_id])

        all_max_comp_leval = np.append(all_max_comp_leval, get_max_depth(block_tree))
        all_avg_comp_leval = np.append(all_avg_comp_leval, get_avg_depth(block_tree))
        all_total_comp_leval = np.append(all_total_comp_leval, get_depth_sum(block_tree))

        all_max_branch = np.append(all_max_branch, get_max_width(block_tree))
        all_avg_branch = np.append(all_avg_branch, get_avg_children(block_tree))
        all_total_branch = np.append(all_total_branch, get_total_branch(block_tree))



    print("#"*15, task, "#"*15)

    print("="*15, "Traditional Metrics", "="*15)

    print('-'*10, "CC", '-'*10)
    print_metric_result(all_score, all_cc, all_loc)

    print('-'*10, "HC(Difficulty)", '-'*10)
    print_metric_result(all_score, all_hc_difficulty, all_loc)

    print('-'*10, "MI", '-'*10)
    print_metric_result(all_score, all_mi, all_loc)

    print('-'*10, "CoC", '-'*10)
    print_metric_result(all_score, all_cog, all_loc)


    print("="*15, "Features", "="*15)

    print('-'*10, "Max CompLevel", '-'*10)
    print_metric_result(all_score, all_max_comp_leval, all_loc, option="partial")

    print('-'*10, "Avg CompLevel", '-'*10)
    print_metric_result(all_score, all_avg_comp_leval, all_loc, option="partial")

    print('-'*10, "Total CompLevel", '-'*10)
    print_metric_result(all_score, all_total_comp_leval, all_loc, option="partial")

    print('-'*10, "Max Branch", '-'*10)
    print_metric_result(all_score, all_max_branch, all_loc, option="partial")

    print('-'*10, "Avg Branch", '-'*10)
    print_metric_result(all_score, all_avg_branch, all_loc, option="partial")

    print('-'*10, "Total Branch", '-'*10)
    print_metric_result(all_score, all_total_branch, all_loc, option="partial")


    print("="*15, "LM-CC", "="*15)
    print_metric_result(all_score, all_lmcc, all_loc, option="partial")
    print("\n")


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

    get_metrics(task, dataset, block_tree_file_path, results_file_path, dataset_output_dir, calculate)

    

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