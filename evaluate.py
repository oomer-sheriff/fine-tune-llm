import argparse
import yaml
import json
from pathlib import Path
from unsloth import FastLanguageModel
from tqdm import tqdm

def evaluate(adapter_path, eval_dataset_path, output_json_path):
    print(f"Loading model with adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    results = []
    total_length = 0
    
    print(f"Running evaluation on {eval_dataset_path}...")
    with open(eval_dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        data = json.loads(line)
        instruction = data['instruction']
        input_text = data.get('input', '')
        
        if input_text:
            prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n"
        else:
            prompt = f"Instruction:\n{instruction}\n\nOutput:\n"
            
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
        
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        results.append({
            "instruction": instruction,
            "input": input_text,
            "expected_output": data.get('output', ''),
            "generated_output": response.strip()
        })
        total_length += len(response.strip())
        
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
        
    print(f"Evaluation complete. Results saved to {output_json_path}")
    if len(results) > 0:
        print(f"Average generated response length (chars): {total_length / len(results):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--output_file", default="evaluation_results.json")
    args = parser.parse_args()
    
    evaluate(args.adapter_path, args.eval_dataset, args.output_file)
