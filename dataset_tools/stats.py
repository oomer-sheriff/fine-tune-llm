import json
import argparse
from transformers import AutoTokenizer

def analyze_dataset(file_path, model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    total_tokens = 0
    max_tokens = 0
    num_examples = 0
    
    lengths = []

    print(f"Analyzing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = f"{data.get('instruction', '')}\n{data.get('input', '')}\n{data.get('output', '')}"
            tokens = tokenizer.encode(text)
            n_tokens = len(tokens)
            
            total_tokens += n_tokens
            max_tokens = max(max_tokens, n_tokens)
            lengths.append(n_tokens)
            num_examples += 1

    if num_examples == 0:
        print("No examples found.")
        return

    lengths.sort()
    p90 = lengths[int(num_examples * 0.9)]
    p99 = lengths[int(num_examples * 0.99)]

    print("\n--- Token Statistics ---")
    print(f"Total examples: {num_examples}")
    print(f"Average tokens per example: {total_tokens / num_examples:.1f}")
    print(f"Max tokens in an example: {max_tokens}")
    print(f"90th percentile token length: {p90}")
    print(f"99th percentile token length: {p99}")
    print("\nRecommendation:")
    if p99 > 1024:
        print("Consider increasing max_seq_length to at least the 99th percentile value, or filtering out long examples.")
    else:
        print("max_seq_length=1024 is sufficient for this dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token lengths of a JSONL dataset")
    parser.add_argument("file_path", help="Path to JSONL file")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", help="Hugging Face model ID for tokenizer")
    args = parser.parse_args()
    analyze_dataset(args.file_path, args.model)
