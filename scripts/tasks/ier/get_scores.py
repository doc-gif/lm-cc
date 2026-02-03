import json
import os
from collections import defaultdict
from typing import Dict, List, Set
from math import comb

def read_json_files(file_paths: List[str]) -> List[Dict]:
    file_contents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            file_contents.append(data)
    
    return file_contents

def calculate_pass_at_k(file_contents: List[Dict], k: int) -> Dict[str, float]:
    id_correct_counts = defaultdict(int)
    all_ids = set()
    
    for file_data in file_contents:
        correct_ids = set(file_data['id_correct'])
        incorrect_ids = set(file_data['id_incorrect'])

        current_file_ids = correct_ids.union(incorrect_ids)
        all_ids.update(current_file_ids)
        
        for id_str in correct_ids:
            id_correct_counts[id_str] += 1
    
    n_files = len(file_contents)
    pass_at_k_scores = {}

    for id_str in all_ids:
        c = id_correct_counts.get(id_str, 0)
        if n_files == 0:
            score = 0.0
        elif c == 0:
            score = 0.0
        else:
            if n_files < k:
                score = 1.0 if c > 0 else 0.0
            else:
                try:
                    score = 1.0 - comb(n_files - c, k) / comb(n_files, k)
                except Exception:
                    score = 0.0

        pass_at_k_scores[id_str] = float(score)

    return pass_at_k_scores



def save_results_to_json(results: Dict[str, float], output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main(k, output_dir, dataset):
    input_files = [
        f"{output_dir}/initial-output/result-{dataset}-1.json",
        f"{output_dir}/initial-output/result-{dataset}-2.json",
        f"{output_dir}/initial-output/result-{dataset}-3.json",
        f"{output_dir}/initial-output/result-{dataset}-4.json",
        f"{output_dir}/initial-output/result-{dataset}-5.json",
    ]

    output_file = f"{output_dir}/results_score.json"
    
    try:

        file_contents = read_json_files(input_files)
        
        pass_at_1_scores = calculate_pass_at_k(file_contents, k)
        
        save_results_to_json(pass_at_1_scores, output_file)
        
    except Exception as e:
        return 1
    
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process HumanEval sample dirs and simplify main.py files.")
    parser.add_argument("--k", type=int, default=1, help="pass@k value to compute (default: 1)")
    parser.add_argument("--output-dir", type=str, default="../output/humaneval-ier", help="")
    parser.add_argument("--dataset", type=str, default="humaneval", help="")

    args = parser.parse_args()
    # k = args.k
    k = 1
    exit_code = main(k, args.output_dir, args.dataset)
    exit(exit_code)