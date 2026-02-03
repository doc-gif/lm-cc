import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from tqdm import tqdm
from create_prompt_ier import create_prompt
from prompt import chatgpt_generator, huggingface_generator, huggingface_generator_chat
import torch
import json
import concurrent.futures
import threading

openai.api_key = os.getenv('OPENAIKEY')
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
MAX_NEW_TOKEN=512
def find_path(dataset, pl, der, model, task):
    root = "../dataset"
    # if dataset == "humaneval_simple":
    #     data_path = os.path.join(root, dataset)
    if not der:
        if dataset in ["CodeNet", "Avatar"]:
            data_path = os.path.join(root, dataset, f"{dataset}-{pl}")
        else:
            data_path = os.path.join(root, dataset)
        if not os.path.exists(data_path):
            print(f'{dataset} does not exist')
    else:
        root_der = f"../Experiment_Results/DER/{task}/"
        if dataset in ["CodeNet", "Avatar"]:
            data_path = os.path.join(root_der, model.split("/")[-1], f"{dataset}-{pl}")
        else:
            data_path = os.path.join(root_der, model.split("/")[-1], dataset)
        if not os.path.exists(data_path):
            print(f'{dataset} does not exist')
    return data_path

def load_model(model_id, cache_dir):
    ## openai models:
    if 'gpt-' in model_id:
        return model_id, None
    ## huggingface models
    if "deepseek-v3" in model_id:
        return model_id, None
    else:
        # print(cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", use_auth_token=AUTH_TOKEN, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cuda", use_auth_token=AUTH_TOKEN, cache_dir=cache_dir)
        
        # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", use_auth_token=AUTH_TOKEN, cache_dir=cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", use_auth_token=AUTH_TOKEN, cache_dir=cache_dir)
        return model, tokenizer


    
def main(model_id, dataset, cache_dir, write_dir, task, pl, der, task_idx=1):
    json_path = "tasks/ier/model_config.json"
    model_config = json.load(open(json_path, 'r'))
    api_type = model_config[model_id]["api"]
    root_dir = find_path(dataset, pl, der, model_id, task)
    if der:
        if dataset in ['CodeNet', 'Avatar']:
            write_root = os.path.join(write_dir, 'DER', task, model_id.split("/")[-1], f"{dataset}-{pl}")
        else:
            write_root = os.path.join(write_dir, 'DER', task, model_id.split("/")[-1], dataset)      
    else:
        if dataset in ['CodeNet', 'Avatar']:
            write_root = os.path.join(write_dir, 'IER', model_id.split("/")[-1], f"{dataset}-{pl}")
        else:
            # write_root = os.path.join(write_dir, 'IER', model_id.split("/")[-1], dataset)
            write_root =  os.path.join(write_dir, f"{dataset}-{task_idx}")
    if not os.path.exists(write_root):
        os.makedirs(write_root)
    model, tokenizer = load_model(model_id, cache_dir)
    hf_lock = threading.Lock()
    max_workers = min(16, (os.cpu_count() or 1) * 4)
    def process_case(d):
        if ".jsonl" in d:
            return
        if pl == 'Java':
            code_path = os.path.join(root_dir, d, 'Main.java')
        else:
            code_path = os.path.join(root_dir, d, 'main.py')
        input_path = os.path.join(root_dir, d, 'input.txt')
        if not os.path.exists(code_path) or not os.path.exists(input_path):
            return
        with open(code_path, 'r') as f:
            code = f.read().strip("\n")
        with open(input_path, 'r') as f:
            code_input = f.read().strip("\n")
        prompt = create_prompt(model_id, code, code_input, dataset, pl)
        try:
            if api_type == "openai":
                err_flg, response = chatgpt_generator(model_id, prompt)
                if err_flg:
                    response = 'ERR: reaches maximum context length'
            elif api_type == "huggingface":
                with hf_lock:
                    response = huggingface_generator((model, tokenizer), prompt, MAX_NEW_TOKEN)
            elif api_type == "huggingface_chat":
                with hf_lock:
                    response = huggingface_generator_chat((model, tokenizer), prompt, MAX_NEW_TOKEN)
            else:
                response = ''
        except Exception as e:
            response = f"ERR: {e}"
        write_folder = os.path.join(write_root, d)
        os.makedirs(write_folder, exist_ok=True)
        write_path = os.path.join(write_folder, 'response.txt')
        with open(write_path, 'w') as wr:
            wr.write(response)

    entries = list(os.listdir(root_dir))
    # entries = entries[:1]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_case, d) for d in entries]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='none')
    parser.add_argument("--dataset", type=str, default='none', help="select one from [CodeNet, Avatar, cruxeval, mbpp, humaneval]")
    parser.add_argument("--cache", type=str, default="./cache")
    parser.add_argument("--writePath", type=str, default="../Experiment_Results")
    parser.add_argument("--task", type=str, default="IER")
    parser.add_argument("--pl", type=str, default="Python", help="select one from [Python, Java]")
    parser.add_argument("--der", help='der inference', action='store_true' )
    parser.add_argument("--task-idx", type=int, default=1, help="")
    args = parser.parse_args()

    model_id = args.model
    dataset = args.dataset
    cache_dir = args.cache
    write_dir = args.writePath
    task = args.task
    pl = args.pl
    der = args.der
    task_idx = args.task_idx
    main(model_id, dataset, cache_dir, write_dir, task, pl, der, task_idx)
