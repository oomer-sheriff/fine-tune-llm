import json
import os
import hashlib
from datasets import load_dataset
import random

def get_hash(instruction, input_text):
    text = f"{instruction.strip()}|||{input_text.strip()}".encode('utf-8')
    return hashlib.sha256(text).hexdigest()

def build_master_dataset():
    os.makedirs("datasets", exist_ok=True)
    
    seen_hashes = set()
    master_data = []
    stats = {"alpaca": 0, "aya": 0, "openorca": 0}

    def add_row(instruction, input_text, output_text, source):
        # Basic validation
        if not instruction or not output_text:
            return False
            
        # Deduplication
        row_hash = get_hash(instruction, input_text)
        if row_hash in seen_hashes:
            return False
            
        seen_hashes.add(row_hash)
        
        row = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output_text.strip()
        }
        master_data.append(row)
        stats[source] += 1
        return True

    # 1. Tamil Alpaca
    print("\nLoading abhinand/tamil-alpaca...")
    alpaca = load_dataset("abhinand/tamil-alpaca", split="train")
    for item in alpaca:
        add_row(
            instruction=item.get("instruction", ""),
            input_text=item.get("input", ""),
            output_text=item.get("output", ""),
            source="alpaca"
        )

    # 2. Cohere Aya Dataset (Filtered for Tamil)
    print("\nLoading CohereForAI/aya_dataset...")
    # Streaming to avoid downloading the entire multilingual dataset
    aya = load_dataset("CohereForAI/aya_dataset", split="train", streaming=True)
    for item in aya:
        if item.get("language") == "tam":
            add_row(
                instruction=item.get("inputs", ""),
                input_text="", # Aya uses 'inputs' as the instruction
                output_text=item.get("targets", ""),
                source="aya"
            )

    # 3. Tamil OpenOrca (Balanced Sample)
    print("\nLoading abhinand/tamil-alpaca-orca...")
    orca = load_dataset("abhinand/tamil-alpaca-orca", split="train")
    orca = orca.shuffle(seed=42).select(range(min(50000, len(orca))))
    
    for item in orca:
        add_row(
            instruction=item.get("instruction", ""),
            input_text=item.get("input", ""),
            output_text=item.get("output", ""),
            source="openorca"
        )

    print("\n" + "="*40)
    print("Shuffling and splitting dataset...")
    random.seed(42)
    random.shuffle(master_data)
    
    # 10% for evaluation/validation, 10% for test (80/10/10 split)
    val_size = int(len(master_data) * 0.10)
    test_size = int(len(master_data) * 0.10)
    
    val_data = master_data[:val_size]
    test_data = master_data[val_size:val_size + test_size]
    train_data = master_data[val_size + test_size:]
    
    train_file = "datasets/master_tamil_train.jsonl"
    val_file = "datasets/master_tamil_val.jsonl"
    test_file = "datasets/master_tamil_test.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for row in train_data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for row in val_data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    with open(test_file, 'w', encoding='utf-8') as f:
        for row in test_data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print("\n🎉 Master Dataset Compilation Complete!")
    print("="*40)
    print(f"Total Unique Rows: {len(master_data)}")
    print(f"Train Split: {len(train_data)} rows -> {train_file}")
    print(f"Val Split:   {len(val_data)} rows -> {val_file}")
    print(f"Test Split:  {len(test_data)} rows -> {test_file}")
    print("\nBreakdown by Source:")
    print(f"- Tamil Alpaca: {stats['alpaca']} rows")
    print(f"- Cohere Aya:   {stats['aya']} rows")
    print(f"- OpenOrca:     {stats['openorca']} rows")

if __name__ == "__main__":
    build_master_dataset()
