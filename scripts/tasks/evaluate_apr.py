import json
import sys
from io import StringIO
from tqdm import tqdm
import re
import os
import numpy as np
import itertools
import timeout_decorator
from multiprocessing import Process, Queue  


def extract_python_code(review_text: str) -> str:
    if review_text is None or review_text == "":
        return ""
    code_blocks = re.findall(r'```python\n(.*?)```', review_text, re.DOTALL)
    if not code_blocks:  
        return ""
    review_text = code_blocks[0].strip()
    return review_text


def estimate_pass_at_k(
    num_samples: int | list[int] | np.ndarray,
    num_correct: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    def estimator(n: int, c: int, k: int):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def execute_with_timeout(code_str, namespace, timeout):
    @timeout_decorator.timeout(timeout, timeout_exception=TimeoutError)
    def inner_exec():
        # print(code_str, file=sys.stderr)
        exec(code_str, namespace)
    inner_exec()

def _run_code_in_subprocess(code_str, test_case, time_limit_sec, result_queue):
    result = {
        'passed': False,
        'actual_output': '',
        'error': ''
    }
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    try:
        sys.stdin = StringIO(test_case['input'])
        sys.stdout = captured_output = StringIO()
        
        namespace = {}
        execute_with_timeout(code_str, namespace, time_limit_sec)
        actual_output = captured_output.getvalue().strip()
        actual_output = [actual_output] if actual_output else []
        result['passed'] = actual_output == test_case['output']
        result['actual_output'] = actual_output
    except TimeoutError:
        result['actual_output'] = "TIME OUT"
    except Exception as e:
        result['actual_output'] = f"Execution Error: {str(e)}"
    finally:
        sys.stdin = original_stdin
        sys.stdout = original_stdout
        result_queue.put(result)

def test_code(code_str, test_cases, time_limit_sec=2):
    results = {
        'passed': 0,
        'failed': 0,
        'details': []
    }

    for i, test_case in enumerate(test_cases, 1):
        result_queue = Queue()
        p = Process(
            target=_run_code_in_subprocess,
            args=(code_str, test_case, time_limit_sec, result_queue)
        )
        p.start()
        p.join(timeout=time_limit_sec + 1)

        if p.is_alive():
            p.terminate()
            p.join()  
            actual_output = "time out"
            passed = False
        else:
            sub_result = result_queue.get()
            passed = sub_result['passed']
            actual_output = sub_result['actual_output']
        if passed:
            results['passed'] += 1
        else:
            results['failed'] += 1
        results['details'].append({
            'test_case': i,
            'input': test_case['input'],
            'expected_output': test_case['output'],
            'actual_output': actual_output,
            'passed': passed
        })

    return results


import argparse
if __name__ == "__main__":
    # TASK = "apr"
    # TASK = "apr_simplemetric"

    parser = argparse.ArgumentParser(description="Run entropy pipeline for Avatar dataset.")
    parser.add_argument("--task", type=str, default="apr", help="")
    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Output Folder to save the API request.",
    )

    args = parser.parse_args()
    TASK = args.task
    llm_gen_path = f"{args.output_dir}/xcodeeval/{TASK}/initial-output"
    data_output_path = f"{args.output_dir}/xcodeeval/{TASK}/python_test_filtered_results.json"
    data_details_output_path = f"{args.output_dir}/xcodeeval/{TASK}/python_test_filtered_results_details.jsonl"



    datas = []
    def extract_first_number(filename):
        return int(filename.split('_')[0])
    files = sorted(os.listdir(llm_gen_path))
    
    print("Evaluate:")

    for file_index, file in enumerate(tqdm(files)):
        data = {}
        
        file_path = os.path.join(llm_gen_path, file)
        with open(file_path, 'r') as f:
            content = json.load(f)
        source_data = content["source_data"]
        
        data["src_uid"] = source_data["src_uid"]
        data["bug_code_uid"] = source_data["bug_code_uid"]
        data["difficulty"] = source_data["difficulty"]
        data["bug_source_code"] = source_data["bug_source_code"]

        # data["task_id"] = file_index
        data["task_id"] = source_data["bug_code_uid"]
        
        results = []
        choices = content["oai_response"]["response"]["choices"]
        total_cnt = len(choices)
        pass_cnt = 0
        pass_rate_list = []
        test_cases = json.loads(source_data["hidden_unit_tests"])
        time_limit_sec = float(source_data["prob_desc_time_limit"].split(' ')[0])
        for idx, c in enumerate(choices):
            # print(f"testing file:", file, "choice:", idx)
            fix_code = extract_python_code(c["message"]["content"])
            test_results = test_code(fix_code, test_cases, time_limit_sec)
            if test_results['failed'] == 0:
                pass_cnt += 1
            pass_rate = test_results['passed'] / (test_results['passed'] + test_results['failed']) if (test_results['passed'] + test_results['failed']) > 0 else 0
            results.append({
                "fix_code": fix_code,
                "is_passed": test_results['failed'] == 0,
                "pass_rate": pass_rate,
                "test_results": test_results
            })
            pass_rate_list.append(pass_rate)
        result = estimate_pass_at_k(
            num_samples=total_cnt,  
            num_correct=[pass_cnt],  
            k=1  
        )
        
        result_3 = estimate_pass_at_k(
            num_samples=total_cnt,  
            num_correct=[pass_cnt],  
            k=3  
        )
        result_5 = estimate_pass_at_k(
            num_samples=total_cnt,  
            num_correct=[pass_cnt],  
            k=5  
        )
        data["pass@1"] = result[0]
        data["pass@3"] = result_3[0]
        data["pass@5"] = result_5[0]
        data["avg_pass_rate"] = np.mean(pass_rate_list) if len(pass_rate_list) > 0 else 0
        data["sample_cnt"] = total_cnt
        data["pass_cnt"] = pass_cnt
        data["test_cnt"] = len(test_cases)
        data["test_results"] = results
        datas.append(data)

    with open(data_details_output_path, 'w') as f:
        for d in datas:
            f.write(json.dumps(d) + '\n')

    results = {}
    for d in datas:
        task_id = d["task_id"]
        results[task_id] = {
            "task_id": task_id,
            "pass@1": d["pass@1"],
            "pass@3": d["pass@3"],
            "pass@5": d["pass@5"],
        }
    with open(data_output_path, 'w') as f:
        f.write(json.dumps(results))


