# Evaluation Guide

## Overview
Evaluating your LoRA adapter ensures it actually learned the instructions.

## Using the Evaluate Script
```bash
python evaluate.py --adapter_path /content/drive/MyDrive/Tamil-LLM/final_adapter --eval_dataset datasets/examples/eval.jsonl --output_file results.json
```

- This script loads your adapter, runs it across the entire `eval_dataset`, and saves the generated outputs to a JSON file.
- It also calculates average token lengths for the responses.

## Using Inference
For interactive testing, use:
```bash
python inference.py --adapter_path /content/drive/MyDrive/Tamil-LLM/final_adapter --instruction "Translate this" --input "Hello World"
```
