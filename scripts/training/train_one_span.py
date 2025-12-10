import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report as seqeval_report

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def create_label_maps_simplified(marker_type):
    label_list = ["O", marker_type]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_list)


def tokenize_and_align_labels_simplified(examples, tokenizer, label_to_id, marker_type):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128,
                                 return_offsets_mapping=True)
    labels = []
    all_markers = examples.get("markers", [])

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        example_labels = [0] * len(offsets)
        example_markers = all_markers[i] if i < len(all_markers) else []

        for marker in example_markers:
            if marker["type"] == marker_type:
                start_char = marker["startIndex"]
                end_char = marker["endIndex"]
                marker_label = label_to_id.get(marker_type)
                if marker_label is not None:
                    for token_idx, (start, end) in enumerate(offsets):
                        if start is not None and end is not None:
                            if start_char <= start < end_char:
                                if token_idx < len(example_labels):
                                    example_labels[token_idx] = marker_label
                            elif start < end_char and end > start_char:
                                if token_idx < len(example_labels) and example_labels[token_idx] == 0:
                                    example_labels[token_idx] = marker_label
        labels.append(example_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    train_file = "train_rehydrated.jsonl"
    model_name = "distilbert-base-uncased"
    output_dir_base = "distilbert-single-type-simplified-baseline"
    batch_size = 32
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    num_epochs = 3
    marker_types_to_train = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    marker_types_to_train = ["Action", "Actor", "Effect", "Evidence", "Victim"]
    all_results = {}

    train_data = load_data(train_file)
    full_dataset = Dataset.from_list(train_data)
    
    train_testvalid = full_dataset.train_test_split(test_size=0.2, seed=42)
    base_train_dataset = train_testvalid['train']
    base_eval_dataset = train_testvalid['test']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for marker_type in marker_types_to_train:
        print(f"\n--- Training BASELINE model for marker type: {marker_type} ---")

        label_to_id, id_to_label, num_labels = create_label_maps_simplified(marker_type)
        
        tokenized_train_dataset = base_train_dataset.map(
            tokenize_and_align_labels_simplified,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id, "marker_type": marker_type}
        )
        
        tokenized_eval_dataset = base_eval_dataset.map(
            tokenize_and_align_labels_simplified,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "label_to_id": label_to_id, "marker_type": marker_type}
        )

        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

        output_dir = f"{output_dir_base}-{marker_type}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir=f'./logs-{marker_type}-simplified',
            report_to="none",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            dataloader_num_workers=0
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print(f"Training simplified model for {marker_type}...")
        trainer.train()
        print(f"Training for {marker_type} finished.")
