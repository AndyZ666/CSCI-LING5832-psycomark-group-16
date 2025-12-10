import json
import argparse
import sys
import os
import zipfile
import torch
from typing import List, Dict, Any
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForTokenClassification

DEFAULT_INPUT_FILE = "dev_rehydrated.jsonl"
DEFAULT_OUTPUT_ZIP = "submission.zip"
TEMP_SUBMISSION_FILE = "submission.jsonl"
BINARY_MODEL_DIR = "distilbert-conspiracy-classification"
SPAN_MODEL_BASE_DIR = "distilbert-single-type-simplified"
MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]

tokenizer = None
binary_model = None
span_models = {}

def load_models():
    """Loads all models into memory."""
    global tokenizer, binary_model, span_models
    
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    print("Loading binary classification model...")
    checkpoints = [d for d in os.listdir(BINARY_MODEL_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        binary_model_path = os.path.join(BINARY_MODEL_DIR, latest_checkpoint)
        binary_model = DistilBertForSequenceClassification.from_pretrained(binary_model_path)
        binary_model.eval()
    else:
        print("Error: No binary model checkpoint found!")
        sys.exit(1)

    print("Loading span extraction models...")
    for marker in MARKER_TYPES:
        model_path = f"{SPAN_MODEL_BASE_DIR}-{marker}"
        try:
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                full_path = os.path.join(model_path, latest)
                model = DistilBertForTokenClassification.from_pretrained(full_path)
                model.eval()
                span_models[marker] = model
                print(f"Loaded {marker} model from {full_path}")
            else:
                print(f"Warning: No checkpoint found for {marker}")
        except FileNotFoundError:
            print(f"Warning: Model directory not found for {marker}")

def predict_conspiracy(text: str) -> str:
    """Runs binary classification inference."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = binary_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
    
    return "Yes" if pred_idx == 1 else "No"

def predict_markers(text: str) -> List[Dict[str, Any]]:
    """Runs span extraction inference for all 5 types."""
    all_markers = []
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    
    for marker_type, model in span_models.items():
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        probs = torch.softmax(logits, dim=2)[0]
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
                
                if start_token < len(offset_mapping) and end_token < len(offset_mapping):
                    char_start = offset_mapping[start_token][0]
                    char_end = offset_mapping[end_token][1]
                    
                    span_text = text[char_start:char_end]
                    
                    if len(span_text) > 2 and char_start != char_end:
                        all_markers.append({
                            "type": marker_type,
                            "startIndex": char_start,
                            "endIndex": char_end,
                            "text": span_text
                        })
            else:
                i += 1
                
    return all_markers

def process_document(item: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = item.get('_id')
    text = item.get('text', '')
    
    if not doc_id or not text:
        return None

    return {
        "_id": doc_id,
        "conspiracy": predict_conspiracy(text),
        "markers": predict_markers(text)
    }

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return data

def save_and_zip(file_path: str, data: List[Dict], output_zip_path: str):
    print(f"Saving submission content to temporary file: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Creating ZIP archive: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=TEMP_SUBMISSION_FILE)

    os.remove(file_path)
    print(f"Successfully created final submission ZIP file: {output_zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default=DEFAULT_INPUT_FILE)
    parser.add_argument("output_zip", nargs='?', default=DEFAULT_OUTPUT_ZIP)
    args = parser.parse_args()

    load_models()

    print(f"Reading data from {args.input_file}...")
    input_data = load_jsonl(args.input_file)
    print(f"Processing {len(input_data)} documents...")

    final_submission = []
    from tqdm import tqdm
    for item in tqdm(input_data):
        res = process_document(item)
        if res:
            final_submission.append(res)

    save_and_zip(TEMP_SUBMISSION_FILE, final_submission, args.output_zip)
