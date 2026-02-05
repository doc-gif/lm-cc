import os
import argparse
import json
from utils.correlation import read_cc, read_lmcc




def evaluate(id2score, id2cc, id2lmcc, simplified_id2score, simplified_id2cc, simplified_id2lmcc):
    simplified_ids = set(simplified_id2score.keys())
    common_ids = [id for id in id2score.keys() if id in simplified_ids]
    assert len(common_ids) == len(simplified_ids)

    id2score = {k: id2score[k] for k in common_ids if k in id2score}
    id2cc = {k: id2cc[k] for k in common_ids if k in id2cc}
    id2lmcc = {k: id2lmcc[k] for k in common_ids if k in id2lmcc}

    print("Original:")
    print("  Score: ", round(sum(id2score.values())/len(id2score), 3))
    print("  CC: ", round(sum(id2cc.values())/len(id2cc), 1))
    print("  LMCC: ", round(sum(id2lmcc.values())/len(id2lmcc), 1))
    print("Simplified:")
    print("  Score: ", round(sum(simplified_id2score.values())/len(simplified_id2score), 3))
    print("  CC: ", round(sum(simplified_id2cc.values())/len(simplified_id2cc), 1))
    print("  LMCC: ", round(sum(simplified_id2lmcc.values())/len(simplified_id2lmcc), 1))





def main(task, dataset_output_dir, simplified_output_dir, results_file_path, simplified_results_file_path, block_tree_file_path, simplified_block_tree_file_path):
    with open(results_file_path, 'r') as f:
        id2score = json.load(f)
    with open(simplified_results_file_path, 'r') as f:
        id2simplifiedscore = json.load(f)
    if task == "program_repair" or task == "code_translation":
        id2score = {k:v["pass@1"] for k, v in id2score.items()}
        id2simplifiedscore = {k:v["pass@1"] for k, v in id2simplifiedscore.items()}

    cc_file_path = os.path.join(dataset_output_dir, 'cc.json')
    simplified_cc_file_path = os.path.join(simplified_output_dir, 'cc.json')
    lmcc_file_path = os.path.join(dataset_output_dir, 'lmcc.json')
    simplified_lmcc_file_path = os.path.join(simplified_output_dir, 'lmcc.json')

    with open(block_tree_file_path, 'r') as f:
        block_trees = json.load(f)
    id2cc = read_cc(cc_file_path, block_trees, False)
    id2lmcc = read_lmcc(lmcc_file_path, block_trees, False)
    
    with open(simplified_block_tree_file_path, 'r') as f:
        simplified_block_trees = json.load(f)
    id2simplifiedcc = read_cc(simplified_cc_file_path, simplified_block_trees, False)
    id2simplifiedlmcc = read_lmcc(simplified_lmcc_file_path, simplified_block_trees, False)

    print("#"*10, f" {task} ", "#"*10)
    evaluate(id2score, id2cc, id2lmcc, id2simplifiedscore, id2simplifiedcc, id2simplifiedlmcc)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    output_dir = "../results"

    THRESHOLD = 0.67
    model_name = "CodeLlama-7b-hf"
    TEMPERATURE = 1.0
    block_tree_filename = f"block_trees-{model_name}-{TEMPERATURE}-thres-{THRESHOLD}.json"

    ### program_repair
    dataset_output_dir = os.path.join(output_dir, "xcodeeval", "apr")
    simplified_output_dir = os.path.join(output_dir, "xcodeeval", "apr-simplified")
    results_file_path = os.path.join(dataset_output_dir, "python_test_filtered_results.json")
    simplified_results_file_path = os.path.join(simplified_output_dir, "python_test_filtered_results.json")
    block_tree_file_path = os.path.join(dataset_output_dir, "entropy", block_tree_filename)
    simplified_block_tree_file_path = os.path.join(simplified_output_dir, "entropy", block_tree_filename)
    main("program_repair", dataset_output_dir, simplified_output_dir, results_file_path, simplified_results_file_path, block_tree_file_path, simplified_block_tree_file_path)

    ### code_translation
    dataset_output_dir = os.path.join(output_dir, "xcodeeval", "code_translation")
    simplified_output_dir = os.path.join(output_dir, "xcodeeval", "code_translation-simplified")
    results_file_path = os.path.join(dataset_output_dir, "python2c_test_filtered_results.json")
    simplified_results_file_path = os.path.join(simplified_output_dir, "python2c_test_filtered_results.json")
    block_tree_file_path = os.path.join(dataset_output_dir, "entropy", block_tree_filename)
    simplified_block_tree_file_path = os.path.join(simplified_output_dir, "entropy", block_tree_filename)
    main("code_translation", dataset_output_dir, simplified_output_dir, results_file_path, simplified_results_file_path, block_tree_file_path, simplified_block_tree_file_path)

    dataset_output_dir = os.path.join(output_dir, "humaneval-ier")
    simplified_output_dir = os.path.join(output_dir, "humaneval-ier-simplified")
    results_file_path = os.path.join(dataset_output_dir, "results_score.json")
    simplified_results_file_path = os.path.join(simplified_output_dir, "results_score_simplified.json")
    block_tree_file_path = os.path.join(dataset_output_dir, "entropy", block_tree_filename)
    simplified_block_tree_file_path = os.path.join(simplified_output_dir, "entropy", block_tree_filename)
    main("execution_reasoning", dataset_output_dir, simplified_output_dir, results_file_path, simplified_results_file_path, block_tree_file_path, simplified_block_tree_file_path)
