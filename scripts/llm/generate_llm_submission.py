import json
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = "gpt-5.1"
INPUT_FILE = "dev_rehydrated.jsonl"
OUTPUT_FILE = "submission.jsonl"
ZIP_FILE = "submission_llm_gpt5.1.zip"

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def extract_spans_with_llm(client, text, model_name):
    prompt = f"""
You are an expert psycholinguist. Extract specific text spans from the Reddit comment below that correspond to these 5 conspiracy markers:
1. Actor (Who is responsible?)
2. Action (What are they doing?)
3. Victim (Who is harmed?)
4. Effect (Consequences?)
5. Evidence (Proof cited?)

Return ONLY a valid JSON object with keys "Actor", "Action", "Victim", "Effect", "Evidence". 
Values must be lists of STRINGS extracted EXACTLY from the text. 
Do not paraphrase. If a marker is missing, use an empty list [].

Text: "{text}"
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\nError processing text: {e}")
        return {"Actor": [], "Action": [], "Effect": [], "Evidence": [], "Victim": []}

def format_for_submission(item_id, text, extraction):
    
    markers = []
    
    for marker_type, spans in extraction.items():
        if not isinstance(spans, list): continue
        
        for span_text in spans:
            if not isinstance(span_text, str) or not span_text: continue
            
            start_idx = text.find(span_text)
            if start_idx != -1:
                end_idx = start_idx + len(span_text)
                markers.append({
                    "type": marker_type,
                    "startIndex": start_idx,
                    "endIndex": end_idx,
                    "text": span_text
                })
    
    submission_id = item_id if item_id.startswith("t1_") else f"t1_{item_id}"
    
    return {
        "_id": submission_id,
        "markers": markers
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="OpenAI API Key")
    args = parser.parse_args()
    
    api_key = args.key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = OpenAI(api_key=api_key)
    
    print(f"Loading data from {INPUT_FILE}...")
    try:
        dev_data = load_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please make sure rehydration was successful.")
        return

    print(f"Loaded {len(dev_data)} examples. Starting inference with {MODEL_NAME}...")
    
    results = []
    
    for item in tqdm(dev_data):
        extraction = extract_spans_with_llm(client, item['text'], MODEL_NAME)
        item_id = item.get('_id', item.get('id'))
        submission_entry = format_for_submission(item_id, item['text'], extraction)
        results.append(submission_entry)

    print(f"Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done! Now zip it: zip {ZIP_FILE} {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
