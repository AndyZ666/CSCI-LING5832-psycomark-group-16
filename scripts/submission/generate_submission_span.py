import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import argparse

def load_data(file_path):
    """Loads data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def predict_spans_for_submission(span_model_base_dir, marker_types, dataset, tokenizer):
    """Runs inference and formats for submission."""
    all_predictions = []
    for item in dataset:
        all_predictions.append({
            "_id": item["_id"],
            "markers": []
        })
    
    for marker in marker_types:
        print(f"\n--- Processing Marker: {marker} ---")
        base_dir = f"{span_model_base_dir}-{marker}"
        
        try:
            checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint")]
            if not checkpoints:
                print(f"Warning: No checkpoints found in {base_dir}, skipping.")
                continue
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            model_path = os.path.join(base_dir, latest_checkpoint)
        except FileNotFoundError:
             print(f"Warning: Directory {base_dir} not found, skipping.")
             continue

        print(f"Loading model from {model_path}...")
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            model.eval()
        except:
            print(f"Warning: Could not load model for {marker}, skipping.")
            continue
            
        print(f"Extracting {marker} spans...")
        for idx, item in enumerate(tqdm(dataset)):
            text = item['text']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=2)[0]
                marker_probs = probs[:, 1].tolist()
            
            threshold = 0.35 
            candidates = [1 if p > threshold else 0 for p in marker_probs]
            
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
                            
                            if char_start == char_end: 
                                continue
                            if char_start == 0 and char_end == 0: 
                                continue
                                
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

if __name__ == "__main__":
    input_file = "dev_rehydrated.jsonl"
    output_file = "submission.jsonl"
    span_model_base_dir = "roberta-single-type-simplified"
    marker_types = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    
    print("--- GENERATING FINAL ROBERTA SUBMISSION ---")
    
    print(f"Loading data from {input_file}...")
    dev_data = load_data(input_file)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    
    results = predict_spans_for_submission(span_model_base_dir, marker_types, dev_data, tokenizer)
    
    print(f"\nSaving submission to {output_file}...")
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Created '{output_file}'. Please zip it into 'submission.zip' and submit.")
