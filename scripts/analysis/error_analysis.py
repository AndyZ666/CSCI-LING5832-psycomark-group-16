import json
from collections import Counter

def load_predictions(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def analyze_binary_errors(data):
    """Analyzes binary classification errors (Conspiracy vs No)."""
    fp = []
    fn = []
    correct = 0
    total = 0
    
    print("\n=== Binary Classification Error Analysis ===")
    
    for item in data:
        true_label = item.get('true_label')
        pred_label = item.get('pred_label')
        
        if true_label == "Can't tell":
            continue
            
        total += 1
        if true_label == pred_label:
            correct += 1
        elif true_label == "No" and pred_label == "Yes":
            fp.append(item)
        elif true_label == "Yes" and pred_label == "No":
            fn.append(item)
            
    print(f"Total Evaluated: {total}")
    print(f"Accuracy: {correct/total:.2%}")
    print(f"False Positives (Alarmist): {len(fp)}")
    print(f"False Negatives (Missed): {len(fn)}")
    
    print("\n--- False Negative Examples (Missed Conspiracies) ---")
    print("Why did the model miss these? (Usually subtle, or long context)")
    for item in fn[:3]:
        print(f"\nID: {item['id']}")
        print(f"Text: {item['text'][:200]}...")
        print(f"True Markers: {[m['type'] for m in item.get('true_markers', [])]}")

    print("\n--- False Positive Examples (False Alarms) ---")
    print("Why did the model overreact? (Usually keywords like 'government', 'lie')")
    for item in fp[:3]:
        print(f"\nID: {item['id']}")
        print(f"Text: {item['text'][:200]}...")

def analyze_span_errors(data):
    """Analyzes span extraction errors."""
    print("\n=== Span Extraction Error Analysis ===")
    
    marker_stats = {
        "Action": {"correct": 0, "missed": 0, "spurious": 0},
        "Actor": {"correct": 0, "missed": 0, "spurious": 0},
        "Effect": {"correct": 0, "missed": 0, "spurious": 0},
        "Evidence": {"correct": 0, "missed": 0, "spurious": 0},
        "Victim": {"correct": 0, "missed": 0, "spurious": 0},
    }
    
    overlap_misses = []
    
    for item in data:
        true_markers = item.get('true_markers', [])
        pred_markers = item.get('pred_markers', [])
        
        for tm in true_markers:
            m_type = tm['type']
            found = False
            for pm in pred_markers:
                if pm['type'] == m_type:
                    if max(tm['startIndex'], pm['startIndex']) < min(tm['endIndex'], pm['endIndex']):
                        found = True
                        break
            
            if found:
                marker_stats[m_type]["correct"] += 1
            else:
                marker_stats[m_type]["missed"] += 1
                
                is_overlap = False
                for other_tm in true_markers:
                    if other_tm == tm: continue
                    if max(tm['startIndex'], other_tm['startIndex']) < min(tm['endIndex'], other_tm['endIndex']):
                        is_overlap = True
                        break
                if is_overlap:
                    overlap_misses.append(item)

        for pm in pred_markers:
            m_type = pm['type']
            found = False
            for tm in true_markers:
                if tm['type'] == m_type:
                    if max(tm['startIndex'], pm['startIndex']) < min(tm['endIndex'], pm['endIndex']):
                        found = True
                        break
            if not found:
                marker_stats[m_type]["spurious"] += 1

    print(f"{'Type':<10} {'Correct':<10} {'Missed':<10} {'Spurious':<10} {'Recall':<10}")
    print("-" * 50)
    for m_type, stats in marker_stats.items():
        recall = stats['correct'] / (stats['correct'] + stats['missed']) if (stats['correct'] + stats['missed']) > 0 else 0
        print(f"{m_type:<10} {stats['correct']:<10} {stats['missed']:<10} {stats['spurious']:<10} {recall:.2%}")

    print(f"\nTotal Overlap-related Misses: {len(overlap_misses)}")
    print("Example of missed overlapping marker:")
    if overlap_misses:
        item = overlap_misses[0]
        print(f"Text: {item['text'][:150]}...")
        print(f"True Markers: {item['true_markers']}")
        print(f"Pred Markers: {item['pred_markers']}")

if __name__ == "__main__":
    preds = load_predictions("baseline_predictions.jsonl")
    if not preds:
        print("No predictions found in baseline_predictions.jsonl")
    else:
        analyze_binary_errors(preds)
        analyze_span_errors(preds)
