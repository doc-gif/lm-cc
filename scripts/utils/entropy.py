import argparse
import os
import sys
from lm_cc.lm_cc import TokenEntropyCalculator, CodeBlockProcessor, get_code_with_boundaries
from .utils import remove_comments_and_docstrings, format_python_code
from tqdm import tqdm
import json

def entropy(dataset, data_path, entropy_output_path, block_tree_output_path):
    if not os.path.exists(os.path.dirname(entropy_output_path)):
        os.makedirs(os.path.dirname(entropy_output_path), exist_ok=True)
    calculator = TokenEntropyCalculator(
        model_name="codellama/CodeLlama-7b-hf",
        cache_dir=cache_dir,
    )
    entropy_contents = []
    if dataset == "humaneval-ier":
        for d in tqdm(os.listdir(data_path)):
            if "__" in d: continue
            task_id = d
            code_path = os.path.join(data_path, d, 'main.py')
            code = open(code_path, 'r').read()
            if not code:
                print("code is None")
            try:
                tokens, entropies = calculator.get_question_entropy(code)
            except Exception as e:
                print(f"Error processing task_id {task_id}:\n {e}")
                continue
            float_entropies = [float(x) for x in entropies] 
            entropy_content = {
                "task_id":task_id,
                "code": code,
                "tokens": tokens,
                "entropies":float_entropies,
            }
            entropy_contents.append(entropy_content)
        with open(entropy_output_path, 'w') as f:
            json.dump(entropy_contents, f, indent=2)

    elif dataset == "xcodeeval-apr":
        files = os.listdir(data_path)
        for _, file in enumerate(tqdm(files)):
            file_path = os.path.join(data_path, file)
            with open(file_path, 'r') as f:
                content = json.load(f)
            source_data = content["source_data"]
            bug_source_code = source_data["bug_source_code"]
            if "using namespace std;" in bug_source_code or "#include" in bug_source_code:
                continue
            bug_source_code = remove_comments_and_docstrings(bug_source_code)
            bug_source_code = format_python_code(bug_source_code)
            if not bug_source_code:
                continue       
            task_id = source_data["bug_code_uid"]
            try:
                tokens, entropies = calculator.get_question_entropy(bug_source_code)
            except Exception as e:
                print(f"Error processing task_id {task_id}:\n {e}")
                continue
            float_entropies = [float(x) for x in entropies] 
            entropy_content = {
                "task_id":task_id,
                "code": bug_source_code,
                "tokens": tokens,
                "entropies":float_entropies,
            }
            entropy_contents.append(entropy_content)
        with open(entropy_output_path, 'w') as f:
            json.dump(entropy_contents, f, indent=2)

    elif dataset == "xcodeeval-ct":
        files = os.listdir(data_path)
        for _, file in enumerate(tqdm(files)):
            file_path = os.path.join(data_path, file)
            with open(file_path, 'r') as f:
                content = json.load(f)
            source_data = content["source_data"]
            source_code = source_data["source_code"]
            if "using namespace std;" in source_code or "#include" in source_code:
                continue
            source_code = format_python_code(source_code)
            source_code = remove_comments_and_docstrings(source_code)
            if not source_code:
                continue
            task_id = source_data["code_uid"]
            try:
                tokens, entropies = calculator.get_question_entropy(source_code)
            except Exception as e:
                print(f"Error processing task_id {task_id}:\n {e}")
                continue
            float_entropies = [float(x) for x in entropies] 
            entropy_content = {
                "task_id":task_id,
                "code": source_code,
                "tokens": tokens,
                "entropies":float_entropies,
            }
            entropy_contents.append(entropy_content)
        with open(entropy_output_path, 'w') as f:
            json.dump(entropy_contents, f, indent=2)

    processor = CodeBlockProcessor()
    for threshold in [0.67]:
        file_contents = []
        for _, d in enumerate(tqdm(entropy_contents)):
            task_id = d["task_id"]
            tokens = d["tokens"]
            entropies = d["entropies"]
            code = d["code"]
            try:
                code_with_boundaries, _, start_end_tokens = get_code_with_boundaries(tokens, entropies, threshold=threshold)
                block_tree_dict = processor.parse_code_blocks(code_with_boundaries, tokens=tokens, start_end_tokens=start_end_tokens)
            except Exception as e:
                print(f"Error processing task_id {task_id}: {e}")
                continue
            content = {
                "task_id":task_id,
                "tokens": tokens,
                "entropies":entropies,
                "code": code,
                "block_tree": block_tree_dict,
            }
            file_contents.append(content)
            if dataset == "humaneval-ier":
                content = {
                    "task_id":f"{task_id}__0",
                    "tokens": tokens,
                    "entropies":entropies,
                    "code": code,
                    "block_tree": block_tree_dict,
                }
                file_contents.append(content)
                content = {
                    "task_id":f"{task_id}__1",
                    "tokens": tokens,
                    "entropies":entropies,
                    "code": code,
                    "block_tree": block_tree_dict,
                }
                file_contents.append(content)

        with open(block_tree_output_path, 'w') as f:
            json.dump(file_contents, f, indent=2)


def main(output_dir, task, cache_dir=""):
    THRESHOLD = 0.67
    model_name = "CodeLlama-7b-hf"
    TEMPERATURE = 1.0
    block_tree_filename = f"block_trees-{model_name}-{TEMPERATURE}-thres-{THRESHOLD}.json"

    
    if task == "program_repair":
        dataset = "xcodeeval-apr"
        dataset_output_dir = os.path.join(output_dir, "xcodeeval", "apr")
        data_path = os.path.join(dataset_output_dir, "initial-output")
        entropy_output_path = os.path.join(dataset_output_dir, "entropy", f"entropies-CodeLlama-7b-hf-{TEMPERATURE}.json")

        # results_file_path = os.path.join(dataset_output_dir, "python_test_filtered_results.json")
    elif task == "code_translation":
        dataset = "xcodeeval-ct"
        dataset_output_dir = os.path.join(output_dir, "xcodeeval", "code_translation")
        data_path = os.path.join(dataset_output_dir, "initial-output")
        entropy_output_path = os.path.join(dataset_output_dir, "entropy", f"entropies-CodeLlama-7b-hf-{TEMPERATURE}.json")
        # results_file_path = os.path.join(dataset_output_dir, "python2c_test_filtered_results.json")
    
    
    elif task == "execution_reasoning":
        dataset = "humaneval-ier"
        dataset_output_dir = os.path.join(output_dir, dataset)
        data_path = "../dataset/humaneval"
        entropy_output_path = os.path.join(dataset_output_dir, "entropy", f"entropies-CodeLlama-7b-hf-{TEMPERATURE}.json")
        # results_file_path = os.path.join(dataset_output_dir, "results_score.json")

    block_tree_output_path = os.path.join(dataset_output_dir, "entropy", block_tree_filename)
    entropy(dataset, data_path, entropy_output_path, block_tree_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entropy pipeline for Avatar dataset.")
    parser.add_argument("--task", type=str, default='program_repair', help="select one from [program_repair, code_translation, execution_reasoning]")
    parser.add_argument("--output-dir", type=str, default="../output")
    parser.add_argument("--model-cache",  type=str, default="", help="model cache directory")
    
    args = parser.parse_args()

    task = args.task
    cache_dir = args.model_cache
    output_dir = args.output_dir
    main(output_dir, task)