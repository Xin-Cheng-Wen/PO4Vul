import numpy as np
from sklearn.metrics import roc_curve
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from collections import defaultdict
def process_detect_vul(value):
    pattern = r'(?<=\nassistant\n)[\s\S]*'


    match = re.search(pattern, value)
    value = match.group()

    if ("NO:" in value):
        return 0
    elif "YES:" in value:
        return 1
    
    elif "YES" in value:
        return 1    
    elif "NO" in value:
        return 0   
    
    else:
x
        return 0  


def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.
    
    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.
    
    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """
    
    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
    
    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()
        
    chosen_threshold = thresholds[idx]
    
    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]
    
    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return vds, chosen_threshold

def calculate_metrics(labels, preds):
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)
    return round(acc,4)*100, round(prec,4)*100, \
        round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
            round(fpr,4)*100, round(fnr,4)*100

def pairwise_predictions(predictions, ground_truth, pred_prob_list):

    pairwise_correct = 0
    pairwise_vulnerable = 0
    pairwise_benign = 0
    pairwise_reversed = 0

    pairwise_correct_probs = 0
    pairwise_vulnerable_probs = 0
    pairwise_benign_probs = 0
    pairwise_reversed_probs = 0

    assert len(predictions) >= 2 and len(ground_truth) >= 2, "输入长度必须至少为2"

    length = min(len(predictions), len(ground_truth))
    if length % 2 != 0:
        length -= 1


    len_pair = len(predictions) // 2
    print(len_pair)

    for i in range(0, len(predictions)-1, 2):
        pred1, pred2 = predictions[i], predictions[i + 1]
        gt1, gt2 = ground_truth[i], ground_truth[i + 1]
        probs1, probs2 = pred_prob_list[i], pred_prob_list[i+1]
        if pred1 == gt1 and pred2 == gt2:

            pairwise_correct += 1
            pairwise_correct_probs = pairwise_correct_probs + (probs1+ probs2)
            
        elif pred1 == 1 and pred2 == 1:
            pairwise_vulnerable += 1
            pairwise_vulnerable_probs = pairwise_vulnerable_probs + (probs1+ probs2)
            # print(str(probs1)+" "+str(probs2))
            # input()
        elif pred1 == 0 and pred2 == 0:
            pairwise_benign += 1
            pairwise_benign_probs = pairwise_benign_probs + (probs1+ probs2)
        elif (pred1 == 1 and pred2 == 0 and gt1 == 0 and gt2 == 1) or (pred1 == 0 and pred2 == 1 and gt1 == 1 and gt2 == 0):
            pairwise_reversed += 1
            pairwise_reversed_probs = pairwise_reversed_probs + (probs1+ probs2)
    dot_number = 2
    return {
        "Pair-wise Correct Prediction (P-C) Number ": pairwise_correct,
        "Pair-wise Correct Prediction (P-C)": round((pairwise_correct / len_pair) * 100, dot_number),
        "Pair-wise Vulnerable Prediction (P-V)": round((pairwise_vulnerable / len_pair) * 100, dot_number),
        "Pair-wise Benign Prediction (P-B)": round((pairwise_benign / len_pair) * 100, dot_number),
        "Pair-wise Reversed Prediction (P-R)": round((pairwise_reversed / len_pair) * 100, dot_number),
        "len_pair":len_pair,
        "A": pairwise_correct_probs/pairwise_correct,
         "B": pairwise_vulnerable_probs/pairwise_vulnerable,
          "C": pairwise_benign_probs/pairwise_benign,
           "D": pairwise_reversed_probs/pairwise_reversed,
    } 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate vulnerability detection score.")
    # parser.add_argument("--pred_file", type=str, help="Path to the file containing model predictions: predictions.txt")
    parser.add_argument("--test_file", type=str, help="Path to the file containing ground truth labels: test.jsonl")
    
    args = parser.parse_args()
    
    # Extract ground truth labels
    with open(args.test_file, 'r') as f:
        lines = f.readlines()
        idx2label = {} # idx to the ground truth label mapping
        response_lines = {}
        label_prob = {}
        category= {}
        
        idx = 0
        for line in lines:
            data = json.loads(line)
            idx2label[idx] = data['target']
            response_lines[idx] = data['response']
            label_prob[idx] = data['label_prob']
            category[idx] = data['category']
            
            idx += 1

    ground_truth = []
    pred = []
    pred_prob_list = []
    category_dict = defaultdict(lambda: {'ground_truth': [], 'pred': []})
    idx = 0
    for sample_key in idx2label:
        ground_truth.append(idx2label[idx])
        response_pred = response_lines[idx]
        
        response_pred = process_detect_vul(response_pred)
        
        pred.append(float(response_pred))
        pred_prob_list.append(float(label_prob[idx]))

        category_dict[category[idx]]['ground_truth'].append(idx2label[idx])
        category_dict[category[idx]]['pred'].append(float(response_pred))
        idx += 1
    
    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr = calculate_metrics(ground_truth, pred)

    result = {
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_tnr": test_tnr,
        "test_fpr": test_fpr,
        "test_fnr": test_fnr,
    }
    results = pairwise_predictions(pred, ground_truth, pred_prob_list)
    print(result)
    print(results)
    
    # Calculate accuracy for each category
    category_accuracy = {}
    all = 0
    for cat, values in category_dict.items():
        cat_ground_truth = values['ground_truth']
        cat_pred = values['pred']
        correct = sum(1 for gt, pr in zip(cat_ground_truth, cat_pred) if gt == pr)
        accuracy = correct / len(cat_ground_truth) if cat_ground_truth else 0
        category_accuracy[cat] = accuracy
        all += len(cat_ground_truth)

    print("Category-wise accuracy:")
    print(category_accuracy)
    for cat, acc in category_accuracy.items():
        print(f"Category: {cat}, Accuracy: {acc:.2f}")
    


    import re

    numbers = re.findall(r'\d+', args.test_file)


    number = int(numbers[0]) if numbers else 0
    rounded_data = {k: round(v, 2) for k, v in category_accuracy.items()}


    output_file_name = 'dpo_v' + str(number) +'.jsonl'
    print(output_file_name)
    
    with open(output_file_name, 'w') as f:
        f.write(json.dumps(rounded_data) + '\n')
