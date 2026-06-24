import os
from pathlib import Path
import argparse
import yaml

def get_latest_checkpoint(drive_path, checkpoint_dir):
    base_dir = Path(drive_path) / checkpoint_dir
    if not base_dir.exists():
        return None
        
    checkpoints = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except ValueError:
                pass
                
    if not checkpoints:
        return None
        
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return str(checkpoints[0][1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", default="configs/train_config.yaml")
    args = parser.parse_args()
    
    with open(args.train_config, 'r') as f:
        config = yaml.safe_load(f)
        
    latest = get_latest_checkpoint(config['storage']['drive_path'], config['storage']['checkpoint_dir'])
    if latest:
        print(latest)
    else:
        print("None")
