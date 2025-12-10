import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm

def load_data(file_path):
    """Loads JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def generate_submission():
    input_file = "dev_rehydrated.jsonl"
    output_file = "submission.jsonl"
    model_dir = "roberta-conspiracy-classification"
    
    print(f"--- Generating Task 2 Submission (Detection) using RoBERTa ---")

    try:
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint")]
        if not checkpoints:
            print("Error: No checkpoints found.")
            return
            
        checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        
        model_path = None
        for cp in checkpoints:
            try:
                path = os.path.join(model_dir, cp)
                print(f"Trying checkpoint: {path}...")
                AutoModelForSequenceClassification.from_pretrained(path)
                model_path = path
                break
            except Exception as e:
                print(f"Checkpoint {cp} is invalid or corrupted: {e}")
                
        if not model_path:
            print("Error: No valid checkpoints found.")
            return
            
        print(f"Using valid model from: {model_path}")
    except Exception as e:
        print(f"Error finding checkpoints: {e}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return
    
    id_to_label = {0: "No", 1: "Yes"}

    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    print(f"Loaded {len(data)} examples from {input_file}")

    if not data:
        print("Error: No data loaded!")
        return

    print("Running inference...")
    batch_size = 16
    results = []
    
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        texts = [item['text'] for item in batch]
        batch_ids = [item['_id'] for item in batch]
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        preds = predictions.cpu().numpy()
        
        for j, pred_idx in enumerate(preds):
            results.append({
                "_id": batch_ids[j], 
                "conspiracy": id_to_label[pred_idx]
            })

    print(f"Submission file generated: {output_file}")
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"Done! Please zip '{output_file}' into 'submission_task2_roberta.zip' and upload to Codabench Task 2!")

if __name__ == "__main__":
    generate_submission()
