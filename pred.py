import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            print(f"[INFO] Prompt length: {len(input_ids)} tokens.")
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0
    current_max_tokens = max_new_tokens
    actual_model_name = model_map.get(model, model)
    
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=actual_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=current_max_tokens,
                extra_body={"top_k": 40},
            )
            
            message = completion.choices[0].message
            content = message.content
            
            if content is None:
                content = getattr(message, 'reasoning_content', None) or ''
            
            return content

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            error_str = str(e)
            
            if "timed out" in error_str.lower():
                print("continue due to timeout")
                continue
                current_max_tokens //= 2
                print(f"Error: Request timed out. Reducing tokens to {current_max_tokens} and retrying (Try {tries}/5)...")
            else:
                print(f"Error Occurs: \"{error_str}\" Retry (Try {tries}/5) ...")
            
            if current_max_tokens < 1:
                break
                
            time.sleep(1)
            
    print("Max tries reached. Failed.")
    return ''

import re
def extract_answer(response):
    # Remove markdown formatting like **A**
    clean_response = response.replace('*', '')
    
    # Look for "Correct Answer" or "The correct answer is" 
    # followed by optional punctuation/space and a single letter A-D
    # Pattern explanation:
    # (?i)        : Case-insensitive
    # \b[A-D]\b   : Matches A, B, C, or D as a whole word
    match = re.search(r'(?:correct answer is|answer:?)\s*\(?([A-D])\)?', clean_response, re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback: Just find the last capitalized A, B, C, or D in the string 
    # if the specific phrase isn't found.
    fallback = re.findall(r'\b([A-D])\b', clean_response)
    return fallback[-1] if fallback else None

def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        # item['prompt'] = prompt

        # if model in model_map:
        #     prompt_ids = tokenizer.encode(prompt)
        # else:
        #     prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        # if len(prompt_ids) < 128000:
        #     print(f"[SKIP] Item {item['_id']}: prompt ({len(prompt_ids)} tokens < 128000).")
        #     continue
        
        # if item["length"] != "short":
            # print("continue")
            # continue
        
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=1.0, max_new_tokens=16384)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=1.0, max_new_tokens=16384)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=1.0, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

        # if item["length"] == "short":
        #     print("breaking")
        #     break

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = data_all
    # for item in data_all:
    #     if item["_id"] not in has_data:
    #         data.append(item)

    # length_order = {"short": 0, "medium": 1, "long": 2}
    # data.sort(key=lambda x: length_order.get(x["length"], 3))

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-5-FP8")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of samples to process (0 = no limit)")
    args = parser.parse_args()
    main()