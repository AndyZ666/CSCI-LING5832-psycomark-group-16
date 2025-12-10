import json
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForTokenClassification
from datasets import Dataset

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

def predict_binary(model_path, dataset, tokenizer):
    """Runs inference for the binary classification task."""
    print(f"Loading binary model from {model_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    predictions = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset['text'][i:i+batch_size]
            inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).tolist()
            predictions.extend(preds)
            
    return predictions

def predict_spans(base_model_path, marker_types, dataset, tokenizer):
    """Runs inference for each span extraction model and combines results."""
    all_span_predictions = [[] for _ in range(len(dataset))]
    
    for marker in marker_types:
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

        print(f"Loading span model for {marker} from {model_path}...")
        try:
            model = DistilBertForTokenClassification.from_pretrained(model_path)
        except:
            print(f"Warning: Could not load model for {marker}, skipping.")
            continue
            
        model.eval()
        
        for idx, text in enumerate(dataset['text']):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)[0].tolist()
            
            
            probs = torch.softmax(logits, dim=2)[0]
            marker_probs = probs[:, 1].tolist()
            
            current_tokens = []
            current_start = -1
            
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            for i, p in enumerate(marker_probs):
                if p > 0.3:
                    if current_start == -1:
                        current_start = i
                    current_tokens.append(tokens[i])
                else:
                    if current_start != -1:
                        end_token_idx = i - 1
                        
                        try:
                            char_start = offset_mapping[current_start][0]
                            char_end = offset_mapping[end_token_idx][1]
                            
                            if char_start != char_end and char_end > char_start:
                                all_span_predictions[idx].append({
                                    "type": marker,
                                    "startIndex": char_start,
                                    "endIndex": char_end,
                                    "text": text[char_start:char_end]
                                })
                        except:
                            pass
                        
                        current_start = -1
                        current_tokens = []
                    
    return all_span_predictions

if __name__ == "__main__":
    val_file = "train_rehydrated.jsonl"
    binary_model_dir = "distilbert-conspiracy-classification"
    span_model_base_dir = "distilbert-single-type-simplified"
    output_file = "baseline_predictions.jsonl"
    
    print("Loading data...")
    all_data = load_data(val_file)
    split_index = int(len(all_data) * 0.8)
    dev_data = all_data[split_index:]
    
    dataset = Dataset.from_list(dev_data)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    print("\n--- Running Binary Classification Inference ---")
    import os
    checkpoints = [d for d in os.listdir(binary_model_dir) if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        binary_model_path = os.path.join(binary_model_dir, latest_checkpoint)
        binary_preds = predict_binary(binary_model_path, dataset, tokenizer)
    else:
        print("No binary model checkpoint found!")
        binary_preds = [0] * len(dataset)

    print("\n--- Running Span Extraction Inference ---")
    marker_types = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    span_preds = predict_spans(span_model_base_dir, marker_types, dataset, tokenizer)
    
    print(f"\nSaving results to {output_file}...")
    id_to_label = {0: "No", 1: "Yes"}
    
    with open(output_file, 'w') as f:
        for i, item in enumerate(dev_data):
            output_item = {
                "id": item["_id"],
                "text": item["text"],
                "true_label": item.get("conspiracy"),
                "pred_label": id_to_label.get(binary_preds[i], "No"),
                "true_markers": item.get("markers", []),
                "pred_markers": span_preds[i]
            }
            f.write(json.dumps(output_item) + "\n")
            
    print("Done! You can now inspect 'baseline_predictions.jsonl'")
