import subprocess
import argparse
from latest_checkpoint import get_latest_checkpoint
import yaml

def resume(model_config, train_config, dataset, val_dataset):
    with open(train_config, 'r') as f:
        config = yaml.safe_load(f)
        
    latest = get_latest_checkpoint(config['storage']['drive_path'], config['storage']['checkpoint_dir'])
    
    cmd = ["python", "train.py", "--model_config", model_config, "--train_config", train_config, "--dataset", dataset]
    if val_dataset:
        cmd.extend(["--val_dataset", val_dataset])
        
    if latest:
        print(f"Found latest checkpoint: {latest}. Resuming...")
        cmd.append("--resume")
    else:
        print("No checkpoints found. Starting fresh training...")
        
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val_dataset", default=None)
    args = parser.parse_args()
    
    resume(args.model_config, args.train_config, args.dataset, args.val_dataset)
