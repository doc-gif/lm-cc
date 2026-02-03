import json
import sys
from io import StringIO
from tqdm import tqdm
import re
import os
import numpy as np
import itertools
import timeout_decorator
import traceback
import tempfile
import subprocess
import resource
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def make_limit_fn(mem_bytes: int, cpu_seconds: int):
    def _limit():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass
    return _limit

def run_c_code(code_str: str, input_str: str, timeout: float, mem_limit_bytes: int = 256 * 1024 * 1024):
    """
    return: {
      "compiled": bool,
      "compile_error": str|None,
      "timeout": bool,
      "stdout": str,
      "stderr": str,
      "rc": int|None,
      "runtime_error": str|None
    }
    """
    res = {
        "compiled": False,
        "compile_error": None,
        "timeout": False,
        "stdout": "",
        "stderr": "",
        "rc": None,
        "runtime_error": None,
    }
    with tempfile.TemporaryDirectory() as td:
        c_path = os.path.join(td, "code.c")
        bin_path = os.path.join(td, "a.out")
        with open(c_path, "w") as f:
            f.write(code_str)
        try:
            comp = subprocess.run(
                ["gcc", "-O2", c_path, "-o", bin_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            if comp.returncode != 0:
                res["compile_error"] = comp.stderr
                return res
            res["compiled"] = True
        except subprocess.TimeoutExpired:
            res["compile_error"] = "compile timeout"
            return res
        except Exception as e:
            res["compile_error"] = str(e)
            return res

        try:
            proc = subprocess.run(
                [bin_path],
                input=input_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                preexec_fn=make_limit_fn(mem_limit_bytes, int(timeout) + 1),
            )
            res["stdout"] = proc.stdout
            res["stderr"] = proc.stderr
            res["rc"] = proc.returncode
            return res
        except subprocess.TimeoutExpired:
            res["timeout"] = True
            return res
        except Exception as e:
            res["runtime_error"] = str(e)
            return res


def test_c_code(code_str, test_cases, time_limit_sec=2):
    results = {"passed": 0, "failed": 0, "details": []}
    for i, test_case in enumerate(test_cases):
    # for i, test_case in enumerate(tqdm(test_cases), 1):
        test_input = test_case.get("input", "")
        expected_output = test_case.get("output", [])
        try:
            r = run_c_code(code_str, test_input, time_limit_sec)
            if not r["compiled"]:
                actual_output = f"COMPILE ERROR: {r['compile_error'] or ''}"
                passed = False
            elif r.get("timeout"):
                actual_output = "TIME OUT"
                passed = False
            elif r.get("runtime_error"):
                actual_output = f"RUNTIME ERROR: {r['runtime_error']}"
                passed = False
            else:
                stdout = (r.get("stdout") or "").strip()
                actual_output = [stdout] if stdout else []
                passed = actual_output == expected_output
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            results["details"].append({
                "test_case": i,
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "passed": passed,
                "compile_error": r.get("compile_error"),
                "stderr": r.get("stderr"),
                "rc": r.get("rc"),
            })
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "test_case": i,
                "input": test_input,
                "expected_output": expected_output,
                "actual_output": f"Execution Error: {str(e)}",
                "passed": False
            })
    return results


def extract_c_code(review_text: str) -> str:
    if review_text is None or review_text == "":
        return ""
    code_blocks = re.findall(r'```c\n(.*?)```', review_text, re.DOTALL)
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
        exec(code_str, namespace)
    inner_exec()

def test_code(code_str, test_cases, time_limit_sec = 2.0):
    results = {
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    namespace = {}
    for i, test_case in enumerate(tqdm(test_cases), 1):
        test_input = test_case['input']
        expected_output = test_case['output']
        
        try:
            original_stdin = sys.stdin
            original_stdout = sys.stdout
            
            sys.stdin = StringIO(test_input)
            sys.stdout = captured_output = StringIO()
            
            passed = False
            actual_output = ""

            try:
                execute_with_timeout(code_str, namespace.copy(), time_limit_sec)
                actual_output = captured_output.getvalue().strip()
                actual_output = [actual_output] if actual_output else []
                passed = actual_output == expected_output
            except TimeoutError:
                print(f"Execution Time Out", file=sys.stderr)
                actual_output="TIME OUT"
                passed = False
            except Exception as e:
                passed = False
                print(f"Execution Error: {e}", file=sys.stderr)

            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1    
            results['details'].append({
                'test_case': i,
                'input': test_input,
                'expected_output': expected_output,
                'actual_output': actual_output,
                'passed': passed
            })
            
        except Exception as e:
            results['failed'] += 1
            results['details'].append({
                'test_case': i,
                'input': test_input,
                'expected_output': expected_output,
                'actual_output': f"Execution Error: {str(e)}",
                'passed': False
            })
        finally:
            sys.stdin = original_stdin
            sys.stdout = original_stdout
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entropy pipeline for Avatar dataset.")
    parser.add_argument("--task", type=str, default="code_translation", help="")
    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Output Folder to save the API request.",
    )
    args = parser.parse_args()
    TASK = args.task

    llm_gen_path = f"{args.output_dir}/xcodeeval/{TASK}/initial-output"
    data_output_path = f"{args.output_dir}/xcodeeval/{TASK}/python2c_test_filtered_results.json"
    data_details_output_path = f"{args.output_dir}/xcodeeval/{TASK}/python2c_test_filtered_results_details.jsonl"

    datas = []
    def extract_first_number(filename):
        return int(filename.split('_')[0])
    files = os.listdir(llm_gen_path)

    
    target_files = files

    max_workers = min(32, (os.cpu_count() or 4))
    lock = threading.Lock()

    def process_file(file_index, filename):
        file_path = os.path.join(llm_gen_path, filename)
        with open(file_path, 'r') as f:
            content = json.load(f)
        source_data = content["source_data"]
        data = {
            "task_id": source_data["code_uid"],
            "src_uid": source_data.get("src_uid"),
            "code_uid": source_data.get("code_uid"),
            "source_code": source_data.get("source_code"),
            "source_lang": source_data.get("source_lang"),
            "target_lang": source_data.get("target_lang"),
            "difficulty": source_data.get("difficulty"),
        }

        results = []
        choices = content["oai_response"]["response"]["choices"]
        total_cnt = len(choices)
        pass_cnt = 0
        pass_rate_list = []
        test_cases = json.loads(source_data["hidden_unit_tests"])
        time_limit_sec = float(source_data["prob_desc_time_limit"].split(' ')[0])

        for idx, c in enumerate(choices):
            translate_code = extract_c_code(c["message"]["content"])
            print(f"Test {file_index} - sample {idx}")
            test_results = test_c_code(translate_code, test_cases, time_limit_sec)

            if test_results['failed'] == 0:
                pass_cnt += 1
            pass_rate = (
                test_results['passed'] / (test_results['passed'] + test_results['failed'])
                if (test_results['passed'] + test_results['failed']) > 0 else 0
            )
            results.append({
                "translate_code": translate_code,
                "is_passed": test_results['failed'] == 0,
                "pass_rate": pass_rate,
                "test_results": test_results
            })
            pass_rate_list.append(pass_rate)

        result = estimate_pass_at_k(num_samples=total_cnt, num_correct=[pass_cnt], k=1)
        result_3 = estimate_pass_at_k(num_samples=total_cnt, num_correct=[pass_cnt], k=3)
        result_5 = estimate_pass_at_k(num_samples=total_cnt, num_correct=[pass_cnt], k=5)

        data.update({
            "pass@1": result[0],
            "pass@3": result_3[0],
            "pass@5": result_5[0],
            "sample_cnt": total_cnt,
            "pass_cnt": pass_cnt,
            "test_cnt": len(test_cases),
            "test_results": results
        })
        return data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, i, fname): fname
            for i, fname in enumerate(target_files)
        }

        for fut in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files"):
            fname = future_to_file[fut]
            try:
                data = fut.result()
                with lock:
                    datas.append(data)
                
            except Exception as e:
                print(f"Error processing {fname}: {e}", file=sys.stderr)
                traceback.print_exc()
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
