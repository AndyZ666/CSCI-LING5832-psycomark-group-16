import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm
import zipfile

MODEL_NAME = "gpt-5.1" 
INPUT_FILE = "dev_rehydrated.jsonl"
OUTPUT_FILE = "submission.jsonl"
ZIP_FILE = "submission_llm_gpt5.1_task2.zip"

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def classify_with_llm(client, text, model_name):
    prompt = f"""
You are an expert psycholinguist specializing in conspiracy theory detection.
Classify whether the following Reddit comment expresses a conspiracy belief.

Task: Binary Classification
Labels: "Yes" (Conspiracy), "No" (Not Conspiracy)

Criteria for "Yes":
- Attributes events to a secret, malevolent actor/group.
- Claims a cover-up or hidden truth.
- Expresses a sense of victimization by powerful forces.
- Rejects mainstream explanations in favor of secret plots.

Text: "{text}"

Return ONLY the label: "Yes" or "No".
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=10
        )
        content = response.choices[0].message.content.strip()
        
        if "yes" in content.lower():
            return "Yes"
        elif "no" in content.lower():
            return "No"
        else:
            return "No"
            
    except Exception as e:
        print(f"\nError processing text: {e}")
        return "No"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="OpenAI API Key")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing", default=None)
    args = parser.parse_args()
    
    api_key = args.key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please provide it via --key or env var.")
        return

    client = OpenAI(api_key=api_key)
    
    print(f"Loading data from {INPUT_FILE}...")
    try:
        dev_data = load_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please make sure rehydration was successful.")
        return

    if args.limit:
        dev_data = dev_data[:args.limit]
        print(f"Limiting to first {args.limit} examples.")

    print(f"Loaded {len(dev_data)} examples. Starting inference with {MODEL_NAME}...")
    
    results = []
    
    for item in tqdm(dev_data):
        label = classify_with_llm(client, item['text'], MODEL_NAME)
        
        item_id = item.get('_id', item.get('id'))
        
        if not item_id.startswith("t1_"):
            item_id = f"t1_{item_id}"
            
        submission_entry = {
            "_id": item_id,
            "conspiracy": label
        }
        results.append(submission_entry)

    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Zipping to {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_FILE)
        
    print(f"Done! Submission ready at {ZIP_FILE}")

if __name__ == "__main__":
    main()
