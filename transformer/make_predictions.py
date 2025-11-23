#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path


COMPONENTS = [
    "ZIPWITH", "*", "MAP", "SQR", "MUL4", "DIV4", "-",
    "MUL3", "DIV3", "MIN", "+", "SCANL", "SHR", "SHL",
    "MAX", "HEAD", "DEC", "SUM", "doNEG", "isNEG",
    "INC", "LAST", "MINIMUM", "isPOS", "SORT", "FILTER",
    "isODD", "REVERSE", "ACCESS", "isEVEN", "COUNT",
    "TAKE", "MAXIMUM", "DROP",
]


def build_io_text(test_set, idx):
    base = Path("../enumerative-search/data") / test_set
    inputs = (base / "input_values.txt").read_text().splitlines()
    outputs = (base / "output_values.txt").read_text().splitlines()

    # DeepCoder example data uses fixed rows for multiple problems
    text = f"Problem {idx}: input={inputs[idx]}, output={outputs[idx]}"
    return text


def write_predictions(output_path, scores):
    with open(output_path, "w") as f:
        for name, s in zip(COMPONENTS, scores):
            f.write(f"{s:.6f} {name}\n")
    print("Wrote:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", default="example")
    parser.add_argument("--problem-idx", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    text = build_io_text(args.test-set, args.problem_idx)
    tokens = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**tokens).logits[0]
        probs = torch.sigmoid(logits)  # multi-label

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{args.problem_idx}.txt"
    write_predictions(out_path, probs.tolist())
