import json
import argparse
import os

def convert(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    if os.path.exists(output_path):
        confirm = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON array from {input_path}: {e}")
            return

    if not isinstance(data, list):
        print("Expected a JSON array (list of objects) in the input file.")
        return

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in data:
            json.dump(item, out_f, ensure_ascii=False)
            out_f.write('\n')
            
    print(f"Successfully converted {len(data)} items to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON array to JSONL format")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    args = parser.parse_args()
    convert(args.input_file, args.output_file)
