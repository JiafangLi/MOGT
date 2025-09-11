import yaml
import os

def print_config(filename, configs):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'a') as f:
        for key, val in configs.items():
            print(key, ':', val, file=f, flush=True)

def if_None(value):
    if isinstance(value, str):
        return None if value == 'None' else value

def parse_gpu(value):
    if isinstance(value, int):
        return f"cuda:{value}"
    return ["cuda:" + gpu.strip() for gpu in value.split(',')]

def convert_config_values(configs, convert_map):
    for key, conversion_fn in convert_map.items():
        if key in configs:
            configs[key] = conversion_fn(configs[key])
    return configs

convert_map = {
    "batch_size": int,
    "cv_folds": int,
    "drop_rate": float,
    "gpu": parse_gpu,
    "heads": int,
    "hidden_dim": int,
    "out_dim":int,
    "lr": float,
    "num_epochs": int,
    "random_seed": int,
    "wandb": bool,
    "stable" : bool,
    "weight_decay": float,
    "edge_threshhold" : int,
    "random" : bool,
    "best" : bool,
    "num_prototypes_per_class" : int,
}

def get():
    with open("config/config.yaml", 'r') as f:
        configs = yaml.safe_load(f)
        configs = convert_config_values(configs, convert_map)
    return configs

if __name__ == "__main__":
    print(get())