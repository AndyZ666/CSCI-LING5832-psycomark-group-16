import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import numpy as np
import os
import argparse

def find_latest_checkpoint(base_dir):
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint")]
    if not subdirs:
        return base_dir
    latest = max(subdirs, key=lambda x: int(x.split("-")[-1]))
    return latest

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def predict_spans(model_base_dir, marker_types, dev_data, tokenizer, use_gap_filling=True):
    all_predictions = [ {"_id": item["_id"], "markers": []} for item in dev_data ]
    
    for marker in marker_types:
        model_dir = f"{model_base_dir}-{marker}"
        model_path = find_latest_checkpoint(model_dir)
        print(f"Loading model for {marker} from {model_path}...")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
        except Exception as e:
            print(f"Failed to load model for {marker}: {e}")
            continue

        print(f"Predicting {marker}...")
        for idx, item in enumerate(tqdm(dev_data)):
            text = item['text']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True, max_length=512)
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                marker_probs = probs[:, 1].tolist()

            threshold = 0.35
            candidates = [1 if p > threshold else 0 for p in marker_probs]

            if use_gap_filling:
                for i in range(1, len(candidates) - 1):
                    if candidates[i-1] == 1 and candidates[i+1] == 1:
                        candidates[i] = 1

            i = 0
            while i < len(candidates):
                if candidates[i] == 1:
                    start_token = i
                    while i < len(candidates) and candidates[i] == 1:
                        i += 1
                    end_token = i - 1
                    
                    try:
                        if start_token < len(offset_mapping) and end_token < len(offset_mapping):
                            char_start = offset_mapping[start_token][0]
                            char_end = offset_mapping[end_token][1]
                            
                            if char_start == char_end: continue
                            if char_start == 0 and char_end == 0: continue
                            
                            span_text = text[char_start:char_end]
                            if len(span_text) > 2:
                                all_predictions[idx]["markers"].append({
                                    "type": marker,
                                    "startIndex": char_start,
                                    "endIndex": char_end,
                                    "text": span_text
                                })
                    except:
                        pass
                else:
                    i += 1
                    
    return all_predictions

def calculate_overlap(pred_start, pred_end, gold_start, gold_end):
    intersection_start = max(pred_start, gold_start)
    intersection_end = min(pred_end, gold_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (pred_end - pred_start) + (gold_end - gold_start) - intersection
    if union == 0: return 0
    return intersection / union

def evaluate(predictions, ground_truth):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    category_metrics = {m: {"tp": 0, "fp": 0, "fn": 0} for m in ["Actor", "Action", "Effect", "Evidence", "Victim"]}

    gt_map = {item["_id"]: item for item in ground_truth}
    
    for pred_item in predictions:
        gt_item = gt_map.get(pred_item["_id"])
        if not gt_item: continue
        
        pred_markers = pred_item["markers"]
        gt_markers = gt_item.get("markers", [])
        
        pred_by_type = {}
        gt_by_type = {}
        
        for m in pred_markers: pred_by_type.setdefault(m["type"], []).append(m)
        for m in gt_markers: gt_by_type.setdefault(m["type"], []).append(m)
        
        for mtype in ["Actor", "Action", "Effect", "Evidence", "Victim"]:
            p_list = pred_by_type.get(mtype, [])
            g_list = gt_by_type.get(mtype, [])
            
            matched_g = set()
            local_tp = 0
            
            for p in p_list:
                best_iou = 0
                best_g_idx = -1
                for i, g in enumerate(g_list):
                    if i in matched_g: continue
                    iou = calculate_overlap(p["startIndex"], p["endIndex"], g["startIndex"], g["endIndex"])
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = i
                
                if best_iou > 0.3:
                    local_tp += 1
                    matched_g.add(best_g_idx)
            
            local_fp = len(p_list) - local_tp
            local_fn = len(g_list) - local_tp
            
            category_metrics[mtype]["tp"] += local_tp
            category_metrics[mtype]["fp"] += local_fp
            category_metrics[mtype]["fn"] += local_fn
            
            total_tp += local_tp
            total_fp += local_fp
            total_fn += local_fn

    f1_sum = 0
    for mtype, counts in category_metrics.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{mtype}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        f1_sum += f1
        
    macro_f1 = f1_sum / 5
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    print(f"\nOverall Macro F1: {macro_f1:.4f}")
    print(f"Overall Micro F1: {micro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    
    print("Loading full train data...")
    full_data = load_data("train_rehydrated.jsonl")
    
    np.random.seed(42)
    indices = np.random.permutation(len(full_data))
    split_idx = int(len(full_data) * 0.8)
    
    val_data = full_data[split_idx:split_idx+200] 
    
    print(f"Running Ablation on {len(val_data)} samples...")
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    model_base = "roberta-single-type-simplified"
    marker_types = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    
    print("\n--- Run 1: WITH Gap Filling (Default) ---")
    preds_with = predict_spans(model_base, marker_types, val_data, tokenizer, use_gap_filling=True)
    f1_with = evaluate(preds_with, val_data)
    
    print("\n--- Run 2: WITHOUT Gap Filling (Ablation) ---")
    preds_without = predict_spans(model_base, marker_types, val_data, tokenizer, use_gap_filling=False)
    f1_without = evaluate(preds_without, val_data)
    
    print(f"\n=== Ablation Results ===")
    print(f"With Gap Filling:    {f1_with:.4f}")
    print(f"Without Gap Filling: {f1_without:.4f}")
    print(f"Impact:              {f1_with - f1_without:+.4f}")
