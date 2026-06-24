import json
import argparse
from pathlib import Path
from collections import defaultdict

def validate_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    valid_lines = 0
    errors = 0
    seen_pairs = set()
    duplicates = 0
    
    lengths = defaultdict(int)

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON")
                errors += 1
                continue
                
            required_keys = ['instruction', 'input', 'output']
            if not all(k in data for k in required_keys):
                print(f"Line {line_num}: Missing required keys. Found: {list(data.keys())}")
                errors += 1
                continue
                
            if not all(isinstance(data[k], str) for k in required_keys):
                print(f"Line {line_num}: Values must be strings.")
                errors += 1
                continue
                
            if not data['output'].strip():
                print(f"Line {line_num}: Empty output.")
                errors += 1
                continue
                
            pair_hash = hash((data['instruction'], data['input']))
            if pair_hash in seen_pairs:
                duplicates += 1
            else:
                seen_pairs.add(pair_hash)

            lengths['instruction'] += len(data['instruction'])
            lengths['input'] += len(data['input'])
            lengths['output'] += len(data['output'])
            
            valid_lines += 1

    print("\n--- Validation Report ---")
    print(f"File: {file_path}")
    print(f"Total valid examples: {valid_lines}")
    print(f"Total errors: {errors}")
    print(f"Total duplicate instruction-input pairs: {duplicates}")
    if valid_lines > 0:
        print(f"Average instruction length (chars): {lengths['instruction'] / valid_lines:.1f}")
        print(f"Average input length (chars): {lengths['input'] / valid_lines:.1f}")
        print(f"Average output length (chars): {lengths['output'] / valid_lines:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSONL dataset for Tamil LLM Fine-tuning")
    parser.add_argument("file_path", help="Path to the JSONL dataset")
    args = parser.parse_args()
    validate_dataset(args.file_path)
