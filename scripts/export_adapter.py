import argparse
from unsloth import FastLanguageModel
import shutil
import os

def export_adapter(checkpoint_path, output_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    
    print(f"Saving adapter to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint folder")
    parser.add_argument("--output", required=True, help="Output path for the standalone adapter")
    args = parser.parse_args()
    export_adapter(args.checkpoint, args.output)
