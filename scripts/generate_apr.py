import os
import sys
import time
import tqdm
import json
import openai
import argparse
import datasets
import concurrent
import numpy as np
from promptsource.templates import Template
import traceback

from openai import OpenAI

SHORT_LANG_MAP = {
    "GNU C++": "C++",
    "GNU C++17": "C++",
    "MS C++ 2017": "C++",
    "MS C++": "C++",
    "Java 8": "Java",
    "Java 6": "Java",
    "GNU C++11": "C++",
    "Java 11": "Java",
    "GNU C++14": "C++",
    "Mono C#": "C#",
    "GNU C": "C",
    "Python 3": "Python",
    "PyPy 3": "Python",
    "GNU C11": "C",
    "Go": "Go",
    "Rust": "Rust",
    "PyPy 2": "Python",
    "Python 2": "Python",
    "MS C#": "C#",
    "Kotlin": "Kotlin",
    "GNU C++0x": "C++",
    "Java 7": "Java",
    "Node.js": "Javascript",
    ".NET Core C#": "C#",
    "PHP": "PHP",
    "GNU C++17 Diagnostics": "C++",
    "Clang++17 Diagnostics": "C++",
    "JavaScript": "Javascript",
    "Ruby": "Ruby",
    "C# 10": "C#",
    "C# 8": "C#",
    "Clang++20 Diagnostics": "C++",
    "GNU C++17 (64)": "C++",
    "GNU C++20 (64)": "C++",
    "Java 17": "Java",
    "Kotlin 1.4": "Kotlin",
    "Kotlin 1.5": "Kotlin",
    "Kotlin 1.6": "Kotlin",
    "Kotlin 1.7": "Kotlin",
    "PyPy 3-64": "Python",
    "Python 3 + libs": "Python",
    "Ruby 3": "Ruby",
    "Rust 2021": "Rust",
}

LANGS = sorted(set([v for k, v in SHORT_LANG_MAP.items()]))




client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY'),  
    base_url = os.getenv('OPENAI_BASE_URL', 'https://qianfan.baidubce.com/v2'),  
)
model_name = os.getenv('OPENAI_MODEL_NAME', 'deepseek-v3')  
def gen(prompt, temperature, nsample):
    cnt = 0
    all_choices = []  
    total_prompt_tokens = 0  
    total_completion_tokens = 0  
    
    while len(all_choices) < nsample:  
        if cnt >= 999:  
            break
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"{prompt}"}],
                temperature=temperature,
                top_p=1,
                n=1,  
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=2048,  
            )
            
            all_choices.extend(response.choices)
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            cnt = 0  
            # print(f"collect {len(all_choices)}/{nsample}")
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"fail")
    
    if not all_choices:
        return None

    response_dict = {
        "id": f"multi-call-{response.id}" if response else "multi-call-unknown", 
        "model": model_name,  
        "created": int(time.time()),  
        "choices": [
            {
                "index": idx,  
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                },
                "finish_reason": choice.finish_reason
            } 
            for idx, choice in enumerate(all_choices)
        ],
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
    }
    
    result = {
        "prompt": prompt,
        "response": response_dict
    }
    return result


xcodeeval_prompt_template = {
    "apr": [
        "Fix a buggy program written in {{lang_cluster}} language to solve the following programming problem:\nDescription: {{prob_desc_description}}\nInput Specification: {{prob_desc_input_spec}}\nOutput Specification: {{prob_desc_output_spec}}\n{% for input, output in zip(prob_desc_sample_inputs, prob_desc_sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nNotes: {{prob_desc_notes}}\nTake input from {{prob_desc_input_from}} and output to {{prob_desc_output_to}}\n\nHere is the code with a bug of {{bug_exec_outcome}}:\n\n{{bug_source_code}}\n\nProvide the fixed {{lang_cluster}} code without any description or extra tokens.\n\nFixed source code:\n ||END-of-SRC|| "
    ]
}


def process_prompt(dt, temperature, template, nsample, output_dir, index, dry_run=0):
    language = dt["lang_cluster"]
    file_path = os.path.join(output_dir, f"{index}_{temperature}_{language}.json")

    # if not os.path.exists(file_path):
    dt["prob_desc_sample_inputs"] = json.loads(dt["prob_desc_sample_inputs"])
    dt["prob_desc_sample_outputs"] = json.loads(dt["prob_desc_sample_outputs"])
    lm_io = template.apply(dt)

    if dry_run:
        open(file_path, "w").write(f"{json.dumps(lm_io[0], indent=4)}")
    else:
        out = gen(lm_io[0], temperature, nsample)
        export_data = {"oai_response": out, "source_data": dt}
        open(file_path, "w").write(f"{json.dumps(export_data, indent=4)}")


def collect_existing_uids(output_dir):
    uids = set()
    try:
        for fn in os.listdir(output_dir):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(output_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
                src = data.get("source_data") if isinstance(data, dict) else None
                if not src:
                    continue
                cid = src.get("bug_code_uid")
                if cid is not None:
                    uids.add(cid)
            except Exception:
                continue
    except Exception:
        pass
    return uids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Output Folder to save the API request.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--nsample",
        default=5,
        type=int,
        help="Number of parallel API request.",
    )
    parser.add_argument(
        "--dataset-path",
        default='../dataset/xcodeeval/apr/xcodeeval_apr_test_filtered.jsonl',
        help="dataset path.",
    )
    parser.add_argument(
        "--incremental",
        type=int,
        default=1,
        help="If 1 (default), skip samples whose code_uid already exists in output-dir json files.",
    )
    args = parser.parse_args()
    out_dir = f"{args.output_dir}/xcodeeval/apr/initial-output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    templates = [
        Template(f"apr_{idx}", template, "xCodeEval")
        for idx, template in enumerate(xcodeeval_prompt_template["apr"])
    ]
    template = templates[0]
    apr_dataset = datasets.load_dataset('json', data_files={'test': args.dataset_path})['test']
    # apr_dataset = apr_dataset.select(range(2))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    incremental = bool(int(args.incremental))
    existing_uids = set()
    if incremental:
        existing_uids = collect_existing_uids(out_dir)
        print(f"Incremental mode ON. Found {len(existing_uids)} existing generated samples in {out_dir}")


    temperature_list = [0.3157894736842105]
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=int(args.num_proc))
    futures = []

    try:
        for idx, dt in tqdm.tqdm(
            enumerate(apr_dataset),
            total=len(apr_dataset),
            desc=f"Preparing samples lang",
        ):
            cid = dt.get("bug_code_uid")
            if incremental and cid in existing_uids:
                continue

            for temperature in temperature_list:
                future = executor.submit(
                    process_prompt,
                    dt,
                    temperature,
                    template,
                    args.nsample,
                    args.output_dir,
                    cid, 
                    args.dry_run,
                )
                futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Calling OpenAI API",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred in task: {e}")
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, shutting down executor and waiting for child processes...")
        executor.shutdown(wait=True, cancel_futures=True)
        print("All child processes have been terminated.")
    
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        traceback.print_exc()
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()

