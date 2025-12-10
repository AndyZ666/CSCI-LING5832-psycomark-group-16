import json
import random
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def get_gold_spans(item):
    """
    Extracts gold spans grouped by type.
    """
    markers = item.get("markers", [])
    gold_spans = {
        "Actor": [],
        "Action": [],
        "Effect": [],
        "Evidence": [],
        "Victim": []
    }
    for m in markers:
        if m["type"] in gold_spans:
            gold_spans[m["type"]].append(m["text"])
    return gold_spans

def evaluate_predictions(gold_spans, predicted_spans):
    """
    Calculates Precision, Recall, F1 based on exact string matching.
    """
    tp = 0
    fp = 0
    fn = 0
    
    for marker_type in gold_spans:
        gold = set([t.lower().strip() for t in gold_spans[marker_type]])
        pred = set([t.lower().strip() for t in predicted_spans.get(marker_type, [])])
        
        tp += len(gold.intersection(pred))
        fp += len(pred - gold)
        fn += len(gold - pred)
        
    return tp, fp, fn

def run_experiment(api_key, num_samples=50):
    client = OpenAI(api_key=api_key)
    
    train_data = load_data("train_rehydrated.jsonl")
    
    if len(train_data) > num_samples:
        sample_data = random.sample(train_data, num_samples)
    else:
        sample_data = train_data
        
    print(f"Running LLM extraction on {len(sample_data)} samples...")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    results = []
    
    for item in tqdm(sample_data):
        text = item['text']
        gold = get_gold_spans(item)
        
        prompt = f"""
You are an expert psycholinguist specializing in conspiracy theory detection.
Your task is to extract specific text spans from the following Reddit comment that correspond to these 5 markers:
1. Actor (Who is allegedly responsible?)
2. Action (What are they doing?)
3. Victim (Who is being harmed?)
4. Effect (What are the negative consequences?)
5. Evidence (What proof is cited?)

Return the output as a JSON object with keys "Actor", "Action", "Victim", "Effect", "Evidence". 
The values should be lists of strings extracted EXACTLY from the text. If no marker is found, use an empty list.

Text: "{text}"
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts information in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                    
                pred_json = json.loads(content)
                
                predicted_spans = {
                    "Actor": pred_json.get("Actor", []),
                    "Action": pred_json.get("Action", []),
                    "Effect": pred_json.get("Effect", []),
                    "Evidence": pred_json.get("Evidence", []),
                    "Victim": pred_json.get("Victim", [])
                }
                
                tp, fp, fn = evaluate_predictions(gold, predicted_spans)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                results.append({
                    "id": item["_id"],
                    "gold": gold,
                    "pred": predicted_spans,
                    "metrics": {"tp": tp, "fp": fp, "fn": fn}
                })
                
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {item['_id']}")
                total_fn += sum(len(v) for v in gold.values())
                
        except Exception as e:
            print(f"API Error: {e}")
            
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nResults on {len(sample_data)} samples:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    with open("llm_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="OpenAI API Key", required=False)
    args = parser.parse_args()
    
    api_key = args.key or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please provide an OPENAI_API_KEY via env var or --key argument.")
    else:
        run_experiment(api_key)
