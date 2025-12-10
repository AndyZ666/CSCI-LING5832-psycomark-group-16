import json
import argparse
import sys
import os
import zipfile
from typing import List, Dict, Any

DEFAULT_INPUT_FILE = "dev_rehydrated.jsonl"
DEFAULT_OUTPUT_ZIP = "submission.zip"
TEMP_SUBMISSION_FILE = "submission.jsonl"

MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]



def predict_conspiracy(text: str) -> str:
    """
    PLACEHOLDER: Replace with actual binary classification model inference logic.
    Loads and runs your 'conspiracy' classification model on the input text.

    Args:
        text: The text content of a single document.

    Returns:
        The predicted label: "Yes" or "No".
    """
    return "No"


def predict_markers(doc_id: str, text: str) -> List[Dict[str, Any]]:
    """
    PLACEHOLDER: Replace with actual span extraction model inference logic.
    Loads and runs FIVE separate models, one for each marker type, and combines their results.

    Args:
        doc_id: The unique ID of the document.
        text: The text content of the document.

    Returns:
        A list of predicted marker dictionaries (with startIndex, endIndex, and type).
    """
    all_markers = []

    for marker_type in MARKER_TYPES:



        pass

    return all_markers


def process_document(item: Dict[str, Any]) -> Dict[str, Any]:
    """Runs all prediction tasks for a single document."""

    doc_id = item.get('_id')
    text = item.get('text', '')

    if not doc_id or not text:
        return None

    conspiracy_pred = predict_conspiracy(text)

    marker_preds = predict_markers(doc_id, text)

    return {
        "_id": doc_id,
        "conspiracy": conspiracy_pred,
        "markers": marker_preds
    }



def load_jsonl(file_path: str) -> List[Dict[str, Any]] | None:
    """Loads all data from a JSONL file."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at {file_path}", file=sys.stderr)
        return None

    print(f"Reading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file_path}: {line.strip()}", file=sys.stderr)
    return data


def save_and_zip(file_path: str, data: List[Dict], output_zip_path: str):
    """Saves the list of dictionaries to a JSONL file and zips it."""

    print(f"Saving submission content to temporary file: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Creating ZIP archive: {output_zip_path}")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=TEMP_SUBMISSION_FILE)

    os.remove(file_path)

    print(f"Successfully created final submission ZIP file: {output_zip_path}")


def parse_args():
    """Parses command-line arguments for file paths."""
    parser = argparse.ArgumentParser(
        description="Generates a combined prediction file (span markers + binary classification) from input data and zips it."
    )

    parser.add_argument(
        "input_file",
        nargs='?',
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input JSONL file (e.g., test or dev set). (Default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "output_zip",
        nargs='?',
        default=DEFAULT_OUTPUT_ZIP,
        help=f"Path to the final output ZIP file containing submission.jsonl. (Default: {DEFAULT_OUTPUT_ZIP})"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    input_data = load_jsonl(args.input_file)

    if input_data is None:
        sys.exit(1)

    print(f"Starting predictions for {len(input_data)} documents...")

    final_submission = []

    for item in input_data:
        prediction_doc = process_document(item)
        if prediction_doc:
            final_submission.append(prediction_doc)

    print(f"Completed predictions for {len(final_submission)} documents.")

    save_and_zip(
        TEMP_SUBMISSION_FILE,
        final_submission,
        args.output_zip
    )
