import jsonlines
import argparse
import json
import sys
from tqdm import tqdm
import traceback
from time import  sleep
import os
import re
import tiktoken


def count_tokens(text):

    encoder = tiktoken.get_encoding("cl100k_base")
    

    tokens = encoder.encode(text)
    

    return len(tokens)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=str, default="None", help='Number')
    args = parser.parse_args()
    filename = './explain_data_train_2.jsonl'

    querys = []
    i = 0
    with jsonlines.open(filename) as reader:
        for obj in reader:
            querys.append(obj)
            i += 1
 



    results = []
    fail = []

    for pos in tqdm(range(len(querys))):
        query = querys[pos]

        code_before = query['code_before']
        code_after = query['code_after']
        code_before_target = "YES: A security vulnerability detected."
        code_after_target = "NO: No security vulnerability."
        code_before_answer = query['answer_before']
        code_after_answer = query['answer_after']



        token_count_before = count_tokens(code_before)
        token_count_after = count_tokens(code_after)

        token_length = 2048
        if token_count_before >= token_length:
            continue
        if token_count_after >= token_length:
            continue



        instruction = f"""You are a security expert that is good at static program analysis. Please analyze the following code and indicate your analysis result with one of the options: 
    (1) YES: A security vulnerability detected.
    (2) NO: No security vulnerability.\n"""
        code_before_reason = code_before_answer.replace("code before the fix", "code")
        code_before_reason = code_before_reason.replace("Code Before", "Code")
        
        code_before_reason = code_before_reason.replace("code before", "code")
        result = {
            "instruction": instruction,
            "input": code_before,
            "output": code_before_target + '\n' + 'Reason:\n' + code_before_reason
        }


        results.append(result)
        code_after_reason = code_after_answer.replace("code after the fix", "code")
        code_after_reason = code_after_reason.replace("Code After", "Code")
        
        code_after_reason = code_after_reason.replace("code After", "code")
        result = {
            "instruction": instruction,
            "input": code_after,
            "output": code_after_target + '\n' + 'Reason:\n' + code_after_reason
        }


        results.append(result)
        instruction = f"""You are a security expert that is good at static program analysis. Please analyze the following vulnerability and corresponding repaired code:\n"""
        result = {
            "instruction": instruction,
            "input": query['filtered_diff_str'],
            "output": ' A security vulnerability detected reson:\n' + code_before_reason + '\n' + 'No security vulnerability reason:\n' + code_after_reason
        }


        results.append(result)

        instruction_loc = f"""You are a security expert that is good at static program analysis. Please identify any security vulnerability detected in the following code, and specify the lines where they triggered or the fixed code.
        (1) YES: A security vulnerability detected.
        (2) NO: No security vulnerability.\n"""

        instruction_ana = f"""You are a security expert that is good at static program analysis. Please identify any security vulnerability detected in the following code, and extract the control flow and data flow step by step for target code.
        (1) YES: A security vulnerability detected.
        (2) NO: No security vulnerability.\n"""

        instruction_int = f"""You are a security expert that is good at static program analysis. Please identify any security vulnerability detected in the following code, and explain why the vulnerability triggered / does not triggered.
        (1) YES: A security vulnerability detected.
        (2) NO: No security vulnerability.\n"""


        code_before_answer = re.sub(r'\[Interpretation\]:\s*\n', '', code_before_reason)

        code_before_answer_matches = re.findall(r'\d+\.\s.*?(?=\n\d+\.|\Z)', code_before_answer, re.DOTALL)
        if code_before_answer_matches:

            cleaned_lines = [re.sub(r'^\d+\.\s*', '', match).strip() for match in code_before_answer_matches]

            if len(cleaned_lines) >= 3:
                code_before_loc, code_before_ana, code_before_int = cleaned_lines[0], cleaned_lines[1], cleaned_lines[2]
                result = {
                    "instruction": instruction_loc,
                    "input": code_before,
                    "output": code_before_target + '\n' + 'Reason:\n' + code_before_loc
                }

                results.append(result)
                result = {
                    "instruction": instruction_ana,
                    "input": code_before,
                    "output": code_before_target + '\n' + 'Reason:\n' + code_before_ana
                }

                results.append(result)
                result = {
                    "instruction": instruction_int,
                    "input": code_before,
                    "output": code_before_target + '\n' + 'Reason:\n' + code_before_int
                }

                results.append(result)
                

            else:
                print("Not enough lines found in the interpretation content.")
        else:
            print("No interpretation content found.")
        

        code_after_answer = re.sub(r'\[Interpretation\]:\s*\n', '', code_after_reason)

        code_after_answer_matches = re.findall(r'\d+\.\s.*?(?=\n\d+\.|\Z)', code_after_answer, re.DOTALL)
        if code_after_answer_matches:
  
            cleaned_lines = [re.sub(r'^\d+\.\s*', '', match).strip() for match in code_after_answer_matches]

            if len(cleaned_lines) >= 3:
                code_after_loc, code_after_ana, code_after_int = cleaned_lines[0], cleaned_lines[1], cleaned_lines[2]
                result = {
                    "instruction": instruction_loc,
                    "input": code_after,
                    "output": code_after_target + '\n' + 'Reason:\n' + code_after_loc
                }
 
                results.append(result)
                result = {
                    "instruction": instruction_ana,
                    "input": code_after,
                    "output": code_after_target + '\n' + 'Reason:\n' + code_after_ana
                }

                results.append(result)
                result = {
                    "instruction": instruction_int,
                    "input": code_after,
                    "output": code_after_target + '\n' + 'Reason:\n' + code_after_int
                }

                results.append(result)
                

            else:
                print("Not enough lines found in the interpretation content.")
        else:
            print("No interpretation content found.")




    new_filename = filename.replace('.jsonl', '_output_all') + args.number + '.json'


    with open(new_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()