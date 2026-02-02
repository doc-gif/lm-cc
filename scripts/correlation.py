import numpy as np
from scipy import stats
import pandas as pd
import pingouin as pg
import argparse
import os
import json
from utils.metrics import get_code_cc, get_code_loc, get_code_cog, get_code_mi, get_code_hc_difficulty, get_lmcc
from utils.utils import get_depth_sum, get_total_branch, get_max_depth, get_max_width, get_avg_children, get_avg_depth

def group_by_metric(score, metric, loc, min_cnt):
    score = np.asarray(score)
    metric = np.asarray(metric)
    if loc is not None:
        loc = np.asarray(loc)

    idx = np.argsort(metric)
    score_sorted = score[idx]
    metric_sorted = metric[idx]
    if loc is not None:
        loc_sorted = loc[idx]

    n = len(score)
    groups = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and (end - start) < min_cnt:
            end += 1
        while end < n and (end - start) < min_cnt:
            end += 1
        groups.append((start, end))
        start = end

    mean_scores = []
    mean_locs = [] if loc is not None else None
    mean_metrics = []
    counts = []
    for s, e in groups:
        mean_scores.append(np.nanmean(score_sorted[s:e]))
        mean_metrics.append(np.nanmedian(metric_sorted[s:e]))
        counts.append(e - s)
        if loc is not None:
            mean_locs.append(np.nanmedian(loc_sorted[s:e]))

    if mean_locs is not None:
        results = list(zip(mean_scores, mean_locs, mean_metrics, counts))
        results_sorted = sorted(results, key=lambda x: (x[2], -x[0]))
        mean_scores_sorted, mean_locs_sorted, mean_metrics_sorted, counts_sorted = zip(*results_sorted)
        mean_scores = list(mean_scores_sorted)
        mean_locs = list(mean_locs_sorted)
        mean_metrics = list(mean_metrics_sorted)
        counts = list(counts_sorted)
    else:
        results = list(zip(mean_scores, mean_metrics, counts))
        results_sorted = sorted(results, key=lambda x: (x[1], -x[0]))
        mean_scores_sorted, mean_metrics_sorted, counts_sorted = zip(*results_sorted)
        mean_scores = list(mean_scores_sorted)
        mean_locs = None
        mean_metrics = list(mean_metrics_sorted)
        counts = list(counts_sorted)
    return mean_scores, mean_locs, mean_metrics, counts

def grouped_partial_corr(score, metric, loc=None, min_cnt = 10):
    score = np.asarray(score, dtype=float)
    metric = np.asarray(metric, dtype=float)

    if loc is None:
        loc_arr = None
    else:
        loc_arr = np.asarray(loc, dtype=float)

    if loc_arr is not None:
        if not (score.shape == loc_arr.shape == metric.shape):
            raise ValueError("score, loc, metric should have the same length")
    else:
        if not (score.shape == metric.shape):
            raise ValueError("score 和 metric should have the same length")

    if score.size == 0:
        raise ValueError("Empty score array!")


    mean_scores, mean_locs, mean_metrics, counts = group_by_metric(
        score, metric, loc_arr, min_cnt
    )

    ms_arr = np.array(mean_scores)
    ml_arr = np.array(mean_locs)
    mm_arr = np.array(mean_metrics)

    counts_arr = np.array(counts, dtype=int)
    if loc_arr is not None:
        base_mask = ~(np.isnan(ms_arr) | np.isnan(ml_arr) | np.isnan(mm_arr))
    else:
        base_mask = ~(np.isnan(ms_arr) | np.isnan(mm_arr))

    valid_mask = base_mask & (counts_arr >= min_cnt)
    valid_count = int(np.count_nonzero(valid_mask))

    spearman_r = spearman_p = np.nan

    if valid_count >= 2:
        x = mm_arr[valid_mask]
        y = ms_arr[valid_mask]

        if loc_arr is None:
            try:
                sr = stats.spearmanr(x, y)
                spearman_r = float(sr.correlation) if hasattr(sr, 'correlation') else float(sr[0])
                spearman_p = float(sr.pvalue) if hasattr(sr, 'pvalue') else float(sr[1])
            except Exception:
                spearman_r = np.nan
                spearman_p = np.nan
        else:
            z = ml_arr[valid_mask]
            df = pd.DataFrame({
                "mean_metric": x,
                "mean_score": y,
                "mean_loc": z,
            })

            # 优先使用 pingouin（常用包），若不可用则回退到残差法
            try:
                spearman = pg.partial_corr(data=df, x='mean_metric', y='mean_score', covar='mean_loc', method='spearman')
                spearman_r = float(spearman['r'].iloc[0])
                spearman_p = float(spearman['p-val'].iloc[0])
            except Exception:
                # 回退实现：对 z 回归，取残差后计算相关（并返回 r 与 p）
                def partial_by_residuals_rp(xv, yv, zv, method='spearman'):
                    # 如果计算 spearman，则先秩变换
                    if method == 'spearman':
                        xv = stats.rankdata(xv)
                        yv = stats.rankdata(yv)
                        zv = stats.rankdata(zv)
                    if np.allclose(zv, zv[0]):
                        return (np.nan, np.nan)
                    coef_x = np.polyfit(zv, xv, 1)
                    resid_x = xv - (coef_x[0] * zv + coef_x[1])
                    coef_y = np.polyfit(zv, yv, 1)
                    resid_y = yv - (coef_y[0] * zv + coef_y[1])
                    if resid_x.size < 2:
                        return (np.nan, np.nan)
                    try:
                        r, p = stats.pearsonr(resid_x, resid_y)
                        return (float(r), float(p))
                    except Exception:
                        return (np.nan, np.nan)
                spearman_r, spearman_p = partial_by_residuals_rp(x, y, z, method='spearman')
    counts = np.array(counts)
    result = {
        "partial_correlation": {
            "spearman-r": None if np.isnan(spearman_r) else float(spearman_r),
            "spearman-pval": None if np.isnan(spearman_p) else float(spearman_p),
        },
        "valid_groups": valid_count,
    }
    return result

def get_grouped_partial_corr(all_scores, all_ccs, all_locs):
    best_result = None
    best_min_cnt = None
    max_abs_spearman_r = 0

    all_scores_len = len(all_scores)
    min_min_cnt = max(1, all_scores_len // 20)
    max_min_cnt = max(1, all_scores_len // 8)

    for min_cnt in range(min_min_cnt, max_min_cnt + 1):
        result = grouped_partial_corr(
            all_scores, all_ccs, loc=all_locs, min_cnt=min_cnt
        )
        valid_groups = result.get("valid_groups", 0)
        spearman = result.get("partial_correlation", {})
        spearman_r = spearman.get("spearman-r", None)
        spearman_pval = spearman.get("spearman-pval", None)
        if (
            8 < valid_groups < 12
            and spearman_r is not None
            and spearman_r < 0
            and spearman_pval is not None
            and spearman_pval < 0.05
        ):
            if abs(spearman_r) > max_abs_spearman_r:
                max_abs_spearman_r = abs(spearman_r)
                best_result = result
                best_min_cnt = min_cnt

    return best_result, best_min_cnt



def read_cc(file_path, block_trees, calculate=True):
    id2cc = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            id2cc[task_id] = get_code_cc(treeinfo["code"])
        with open(file_path, 'w') as f:
            json.dump(id2cc, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2cc = json.load(f)
    return id2cc

def read_loc(file_path, block_trees, calculate=True):
    id2loc = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            id2loc[task_id] = get_code_loc(treeinfo["code"])
        with open(file_path, 'w') as f:
            json.dump(id2loc, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2loc = json.load(f)
    return id2loc

def read_cog(file_path, block_trees, calculate=True):
    id2cog = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            id2cog[task_id] = get_code_cog(treeinfo["code"])
        with open(file_path, 'w') as f:
            json.dump(id2cog, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2cog = json.load(f)
    return id2cog

def read_mi(file_path, block_trees, calculate=True):
    id2mi = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            id2mi[task_id] = get_code_mi(treeinfo["code"])
        with open(file_path, 'w') as f:
            json.dump(id2mi, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2mi = json.load(f)
    return id2mi
        
def read_hc_difficulty(file_path, block_trees, calculate=True):
    id2hc_difficulty = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            id2hc_difficulty[task_id] = get_code_hc_difficulty(treeinfo["code"])
        with open(file_path, 'w') as f:
            json.dump(id2hc_difficulty, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2hc_difficulty = json.load(f)
    return id2hc_difficulty

def read_lmcc(file_path, block_trees, calculate=True):
    id2lmcc = {}
    if calculate or not os.path.exists(file_path):
        for treeinfo in block_trees:
            task_id = f"{treeinfo['task_id']}"
            block_tree = treeinfo["block_tree"]
            id2lmcc[task_id] = get_lmcc(block_tree)
        with open(file_path, 'w') as f:
            json.dump(id2lmcc, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            id2lmcc = json.load(f)
    return id2lmcc


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

def get_metrics(output_dir, dataset, block_tree_file_path, results_file_path, dataset_output_dir):
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

    get_metrics(output_dir, dataset, block_tree_file_path, results_file_path, dataset_output_dir)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default='xcodeeval-apr', help="select one from [xcodeeval-apr, xcodeeval-ct, humaneval-ier]")
    parser.add_argument("--task", type=str, default='program_repair', help="select one from [program_repair, code_translation, execution_reasoning]")
    parser.add_argument("--output-dir", type=str, default="../output")
    parser.add_argument("--calculate", action='store_true')
    args = parser.parse_args()

    task = args.task
    output_dir = args.output_dir
    calculate = args.calculate
    main(task, output_dir, calculate)