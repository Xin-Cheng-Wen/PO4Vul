import numpy as np
from sklearn.metrics import roc_curve
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def process_detect_vul(value):

    value_small = value.lower()
    if ("NO:" in value):
        return 0
    elif "YES:" in value:
        return 1

    elif "vulnerability in target code" in value:
        return 1   
    elif "vulnerability in the target code" in value:
        return 1 
    elif "vulnerability in the original code " in value:
        return 1 
        
    elif "fixed code in target code " in value:
        return 0 
    elif "No vulnerability" in value:
        return 0 
    elif "YES" in value:
        return 1    
  
    elif "NO" in value:
        return 0   
    
    else:

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

def pairwise_predictions(predictions, ground_truth):
    
    pairwise_correct = 0
    pairwise_vulnerable = 0
    pairwise_benign = 0
    pairwise_reversed = 0

    
    assert len(predictions) >= 2 and len(ground_truth) >= 2, "len() >= 2"

    
    length = min(len(predictions), len(ground_truth))
    if length % 2 != 0:
        length -= 1


    len_pair = len(predictions) // 2
    print(len_pair)
    
    for i in range(0, len(predictions)-1, 2):
        pred1, pred2 = predictions[i], predictions[i + 1]
        gt1, gt2 = ground_truth[i], ground_truth[i + 1]

        if pred1 == gt1 and pred2 == gt2:
            pairwise_correct += 1
        elif pred1 == 1 and pred2 == 1:
            pairwise_vulnerable += 1
        elif pred1 == 0 and pred2 == 0:
            pairwise_benign += 1
        elif (pred1 == 1 and pred2 == 0 and gt1 == 0 and gt2 == 1) or (pred1 == 0 and pred2 == 1 and gt1 == 1 and gt2 == 0):
            pairwise_reversed += 1
    dot_number = 2
    return {
        "Pair-wise Correct Prediction (P-C) Number ": pairwise_correct,
        "Pair-wise Reversed Prediction (P-R) Number ": pairwise_reversed,
        "Pair-wise Correct Prediction (P-C)": round((pairwise_correct / len_pair) * 100, dot_number),
        "Pair-wise Vulnerable Prediction (P-V)": round((pairwise_vulnerable / len_pair) * 100, dot_number),
        "Pair-wise Benign Prediction (P-B)": round((pairwise_benign / len_pair) * 100, dot_number),
        "Pair-wise Reversed Prediction (P-R)": round((pairwise_reversed / len_pair) * 100, dot_number),
        "len_pair":len_pair
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
        idx = 0
        for line in lines:
            data = json.loads(line)
            idx2label[idx] = data['target']
            response_lines[idx] = data['response']
            idx += 1

    ground_truth = []
    pred = []
    idx = 0
    for sample_key in idx2label:
        print(idx)
        ground_truth.append(idx2label[idx])
        response_pred = response_lines[idx]
        
        response_pred = process_detect_vul(response_pred)
        
        pred.append(float(response_pred))
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
    results = pairwise_predictions(pred, ground_truth)
    print(result)
    print(results)


    
