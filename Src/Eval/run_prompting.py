import argparse
import os
import time
import json
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT

import tiktoken

# add your OpenAI API key here



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
        encoding = tiktoken.encoding_for_model(model)
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
                print(f"Truncating message: {value[:100]}...")
                tm[key] = encoding.decode(encoded_value[:max_tokens - num_tokens])
                break
            else:
                tm[key] = value
        trunc_messages.append(tm)
    return trunc_messages


# get completion from an OpenAI chat model
def get_openai_chat(
    prompt,
    args
):
    if args.fewshot_eg:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
            {"role": "user", "content": prompt["prompt"]}
        ]
    else:
        # select the correct in-context learning prompt based on the task
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
            ]
    
    # count the number of tokens in the prompt
    messages = truncate_tokens_from_messages(messages, args.model, args.max_gen_length)

    
    # print(messages)
    try:
        client = OpenAI(api_key="EMPTY",base_url="http://localhost:8000/v1")
        # messages = [{"role": "user", "content": "Who are you?"}]
        result = client.chat.completions.create(messages=messages, model ="Qwen2.5-Coder-7B-Instruct")
        response_content = str(result.choices[0].message.content)
        print(response_content)
    

            

    
        return response_content

    # when encounter RateLimit or Connection Error, sleep for 5 or specified seconds and try again
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as error:
        retry_time = error.retry_after if hasattr(error, "retry_after") else 5
        print(f"Rate Limit or Connection Error. Sleeping for {retry_time} seconds ...")
        time.sleep(retry_time)
        return get_openai_chat(
            prompt,
            args,
        )
    # when encounter bad request errors, print the error message and return None
    except openai.BadRequestError as error:
        print(f"Bad Request Error: {error}")
        return None

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
        prompts.append(p)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="saves/qwn2.5_vul/full/sft_6", choices=["saves/qwn2.5_vul/full/sft_6", "gpt-3.5-turbo-0125", "gpt-4-0125-preview", "gpt-4-32k"], help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="standard", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    args = parser.parse_args()

    output_file = os.path.join(args.output_folder, f"{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}.jsonl")
    if args.prompt_strategy == "std_cls":
            inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    with open(output_file, "a") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} prompts ...")
        idx = 1 
        for p in tqdm(prompts):

            response = get_openai_chat(p, args)
            
            if response is None:
                response = "ERROR"
            p["idx"] =idx
            p["response"] = response
            f.write(json.dumps(p))
            f.write("\n")
            f.flush()
            idx += 1


if __name__ == "__main__":
    main()