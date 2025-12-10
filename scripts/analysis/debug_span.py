import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import numpy as np

def debug_span_model():
    marker_type = "Action"
    model_path = f"distilbert-single-type-simplified-{marker_type}/checkpoint-2160"
    
    print(f"--- Debugging {marker_type} Model ---")
    try:
        model = DistilBertForTokenClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    text = "Deep State has been using Stefan A. Halper to spy on presidential campaigns and infiltrate them since Carter"
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print(f"\nText: {text}")
    print("\nToken predictions (Prob(O) vs Prob(Marker)):")
    print(f"{'Token':<15} {'Prob(O)':<10} {'Prob(Action)':<10} {'Pred'}")
    print("-" * 50)
    
    for i, token in enumerate(tokens):
        prob_o = probs[0][i][0].item()
        prob_marker = probs[0][i][1].item()
        pred = "Action" if prob_marker > prob_o else "O"
        
        if prob_marker > 0.01:
            print(f"{token:<15} {prob_o:.4f}     {prob_marker:.4f}           {pred}")

if __name__ == "__main__":
    debug_span_model()
