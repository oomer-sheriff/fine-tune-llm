import json
import os

def download():
    print("Downloading abhinand/tamil-alpaca from Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install 'datasets': pip install datasets")
        return
        
    ds = load_dataset("abhinand/tamil-alpaca", split="train[:10]")
    
    os.makedirs("datasets/examples", exist_ok=True)
    out_path = "datasets/examples/sample.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            obj = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"Saved 10 examples to {out_path}")

if __name__ == "__main__":
    download()
