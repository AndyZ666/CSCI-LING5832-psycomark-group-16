import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_and_filter_data(file_path):
    """Loads data from a JSON file and filters out entries with 'Can't tell'."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'conspiracy' in item and item['conspiracy'] in ["Yes", "No"]:
                    data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data."""
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)


def encode_labels(dataset, label_to_id):
    """Encodes the labels to numerical values."""
    return dataset.map(lambda examples: {'labels': [label_to_id[label] for label in examples["conspiracy"]]},
                       batched=True)


def save_predictions(trainer, test_dataset, output_file):
    """Saves predictions and true labels to a JSON file."""
    predictions = trainer.predict(test_dataset)
    predicted_classes = np.argmax(predictions.predictions, axis=-1)
    true_labels = test_dataset["labels"]
    results = []
    for i in range(len(true_labels)):
        results.append({
            "predicted_label": trainer.model.config.id2label[predicted_classes[i]],
            "true_label": trainer.model.config.id2label[true_labels[i]],
            "text": test_dataset["text"][i]
        })
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    train_file = "train_rehydrated.jsonl"
    model_name = "roberta-base"
    output_dir = "roberta-conspiracy-classification"
    label_to_id = {"No": 0, "Yes": 1}
    id_to_label = {0: "No", 1: "Yes"}
    num_labels = len(label_to_id)
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 5

    train_data = load_and_filter_data(train_file)
    
    full_dataset = Dataset.from_list(train_data)
    
    train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_testvalid['train']
    eval_dataset = train_testvalid['test']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_train_dataset = tokenize_data(train_dataset, tokenizer)
    encoded_train_dataset = encode_labels(tokenized_train_dataset, label_to_id)
    
    tokenized_eval_dataset = tokenize_data(eval_dataset, tokenizer)
    encoded_eval_dataset = encode_labels(tokenized_eval_dataset, label_to_id)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id_to_label,
                                                                label2id=label_to_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir='./logs_roberta',
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Training the model...")
    trainer.train()
    print("Training finished.")
