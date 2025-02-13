import jsonlines
import json
import sys
from tqdm import tqdm
import traceback
from time import  sleep
import os
import re
import tiktoken
import random
import argparse
# print(1)
# exit(0)
filename = './explain_data_train_2.jsonl'
cwe_dict = {0: ['119', '125', '787', '120', '122', '131', '121'], 1: ['20', '129', '241'], 2: ['401', '772', '404', '399'], 3: ['416', '415', '763', '672'], 4: ['190', '189'], 5: ['476'], 6: ['287', '307', '294', '288', '290', '254'], 7: ['400', '834', '674', '770'], 8: ['362', '667', '361'], 9: ['909', '665', '824', '908', '457'], 10: ['200', '703', '369', '284', '835', '617', '22', '79', '269', '704', '89', '78', '754', '59', '327', '94', '732', '552', '276', '295', '61', '134', '682', '697', '77', '863', '707', '532', '193', '755', '345', '74', '862', '601', '252', '843', '88', '502', '281', '347', '285', '611', '346', '191', '273', '349', '311', '352', '668', '203', '918', '639', '693', '1021', '93', '681', '116', '354', '172', '522', '664', '212', '823', '113', '798', '326', '670', '331', '426', '444', '613', '209', '264', '310', '19', '17', '388', '320', '16', '417', '255']}
       
def count_tokens(text):

    encoder = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoder.encode(text)
    

    return len(tokens)
querys = []
i = 0

with jsonlines.open(filename) as reader:
    for obj in reader:
        querys.append(obj)
        i += 1



results = []
fail = []

def extract_number(cwe_string):
    match = re.search(r'\d+', cwe_string)
    return int(match.group()) if match else None

def find_key(cwe_dict, value):
    for key, values in cwe_dict.items():
        if str(value) in values:
            return key

    return 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate vulnerability detection score.")
    # parser.add_argument("--pred_file", type=str, help="Path to the file containing model predictions: predictions.txt")
    parser.add_argument("--number", type=int, help="improve number")
    
    args = parser.parse_args()


    with open('./output_type_dir/dpo_v' + str(args.number) +'.jsonl', 'r') as f:
        line = f.readline()
        categories = json.loads(line)

    print(categories)



    for pos in tqdm(range(len(querys))):
        

        query = querys[pos]
        cwe_id = query['cwe_id'][0]
        # cwe_id = p['cwe'][0]        
        if cwe_id != None:
            cwe = extract_number(cwe_id)
            category = find_key(cwe_dict, cwe)
        probabilities = {category: 1 - accuracy for category, accuracy in categories.items()}
        

        random_number = random.random()
        if random_number >= probabilities[str(category)]:

            continue

        code_before = query['code_before']
        code_after = query['code_after']
        code_before_target = "YES: A security vulnerability detected."
        code_after_target = "NO: No security vulnerability."
        code_before_answer = query['answer_before']
        code_after_answer = query['answer_after']


        code_before_reason = code_before_answer
        code_after_reason = code_after_answer
        token_count_before = count_tokens(code_before)
        token_count_after = count_tokens(code_after)

        token_length = 2048
        if token_count_before >= token_length:
            continue
        if token_count_after >= token_length:
            continue
        
        instruction = f"""You are a security expert that is good at static program analysis. Please analyze the following code and indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability."""

        code_before_reason = code_before_reason.replace("code before the fix", "code")
        code_before_reason = code_before_reason.replace("Code Before", "Code")
        
        code_before_reason = code_before_reason.replace("code before", "code")

        code_after_reason = code_after_reason.replace("code after the fix", "code")
        code_after_reason = code_after_reason.replace("Code After", "Code")
        
        code_after_reason = code_after_reason.replace("code After", "code")

        result = {
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction + '\n' + code_after
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value":code_after_target + '\n' + 'Reason:\n' + code_after_reason
                },
                "rejected": {
                    "from": "gpt",
                    "value": code_before_target + '\n' + 'Reason:\n' + code_before_reason
                }
            }
        


        results.append(result)

        result = {
                "conversations": [
                    {
                        "from": "human",
                        "value": instruction + '\n' + code_before
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": code_before_target + '\n' + 'Reason:\n' + code_before_reason
                },
                "rejected": {
                    "from": "gpt",
                    "value":code_after_target + '\n' + 'Reason:\n' + code_after_reason
                }
            }
        


        results.append(result)



    new_filename = filename.replace('.jsonl', '_dpo_online_v1') + '.json'


    with open(new_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
