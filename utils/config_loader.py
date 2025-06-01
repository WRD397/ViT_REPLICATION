import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config_path = '/config/vit_test_config.yaml'
    config = load_config(config_path)
    print("Loaded config:")
    print(config)