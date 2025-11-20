import yaml

def load_yaml(load_path):
    with open(load_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(config, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)