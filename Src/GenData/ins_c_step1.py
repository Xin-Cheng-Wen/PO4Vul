from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 加载模型和分词器
import tiktoken
import argparse
import os
import time
import json
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN
import re
from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT


def truncate_tokens_from_messages(messages, model, max_gen_length):
    """
    Count the number of tokens used by a list of messages, 
    and truncate the messages if the number of tokens exceeds the limit.
    Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    if model == "gpt-3.5-turbo-0125":
        max_tokens = 16385 - max_gen_length
    elif model == "gpt-4-32k":
        max_tokens = 128000 - max_gen_length
    else:
        max_tokens = 4096 - max_gen_length
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3

    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    trunc_messages = []
    for message in messages:
        tm = {}
        num_tokens += tokens_per_message
        for key, value in message.items():
            
            encoded_value = encoding.encode(value)
            num_tokens += len(encoded_value)
            if num_tokens > max_tokens:
                # print(f"Truncating message: {value[:100]}...")
                tm[key] = encoding.decode(encoded_value[:max_tokens - num_tokens])
                break
            else:
                tm[key] = value
        trunc_messages.append(tm)
    return trunc_messages

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        commit_id = sample["commit_id"]
        p = {"commit_id":commit_id}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        
        p["cwe"] = sample["cwe"]

        prompts.append(p)
    return prompts

def extract_number(cwe_string):
    match = re.search(r'\d+', cwe_string)
    return int(match.group()) if match else None

def find_key(cwe_dict, value):
    for key, values in cwe_dict.items():
        if str(value) in values:
            return key
    print(value)
    return "Value not found in dictionary"

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", choices=["Qwen/Qwen2.5-Coder-7B-Instruct", "gpt-3.5-turbo-0125", "gpt-4-0125-preview", "gpt-4-32k"], help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="cot", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, default = './primevul_valid_paired.jsonl', help='Data path')
    parser.add_argument('--output_folder', type=str, default = './output_type_dir', help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--number', type=int)
    args = parser.parse_args()
    
    output_file = os.path.join(args.output_folder, f"{args.data_path}_{args.prompt_strategy}_test_v{args.number}.jsonl")
    
    device = "cuda" 
    model_name = "/saves/qwn2.5_vul/full/sft_6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    cwe_dict = {0: ['119', '125', '787', '120', '122', '131', '121'], 1: ['20', '129', '241'], 2: ['401', '772', '404', '399'], 3: ['416', '415', '763', '672'], 4: ['190', '189'], 5: ['476'], 6: ['287', '307', '294', '288', '290', '254'], 7: ['400', '834', '674', '770'], 8: ['362', '667', '361'], 9: ['909', '665', '824', '908', '457'], 10: ['200', '703', '369', '284', '835', '617', '22', '79', '269', '704', '89', '78', '754', '59', '327', '94', '732', '552', '276', '295', '61', '134', '682', '697', '77', '863', '707', '532', '193', '755', '345', '74', '862', '601', '252', '843', '88', '502', '281', '347', '285', '611', '346', '191', '273', '349', '311', '352', '668', '203', '918', '639', '693', '1021', '93', '681', '116', '354', '172', '522', '664', '212', '823', '113', '798', '326', '670', '331', '426', '444', '613', '209', '264', '310', '19', '17', '388', '320', '16', '417', '255']}
       


    inst = PROMPT_INST_COT
    prompts = construct_prompts(args.data_path, inst)


    with open(output_file, "a") as f:
        idx = 1 
        for p in tqdm(prompts):

            
            cwe_id = p['cwe'][0]
            if cwe_id != None:
                cwe = extract_number(cwe_id)
                category = find_key(cwe_dict, cwe)
                p["category"] = category
            else: 
                p["category"] = 11
            
            
            messages = [
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": p["prompt"]}
            ]
            messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)


            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize the text and create the attention mask
            model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            attention_mask = model_inputs['attention_mask']

            # Generate outputs with the attention mask
            with torch.no_grad():
                outputs = model.generate(
                    model_inputs['input_ids'],
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=4096
                )
            decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            # print("Generated text:", decoded_output)

            logits = outputs.scores
            probs = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]

            generated_tokens = outputs.sequences[0][len(model_inputs['input_ids'][0]):]
            yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            yes_token_id_1 = tokenizer.encode("YES", add_special_tokens=False)[0]
            no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
            no_token_id_1 = tokenizer.encode("NO", add_special_tokens=False)[0]
            for i, token in enumerate(generated_tokens):
                if token == yes_token_id or token == no_token_id or token == yes_token_id_1 or token == no_token_id_1:
                    token_str = tokenizer.decode(token)
                    token_prob = probs[i][0, token].item()
                    break
                    # print(f"Token: {token_str}, Probability: {token_prob:.4f}")
            p["label_prob"] = token_prob
            p["label_pred"] = token_str
            p["response"] = decoded_output
            '''
            tokens_with_probs = []
            for i, token in enumerate(generated_tokens):
                token_str = tokenizer.decode(token)
                token_prob = probs[i][0, token].item()
                tokens_with_probs.append((token_str, token_prob))
                # print(f"Token: {token_str}, Probability: {token_prob:.4f}")
                #         f.write(json.dumps(p))
            p["token_probs"] = tokens_with_probs
            '''
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()
            idx += 1
if __name__ == "__main__":
    main()