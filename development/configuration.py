import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Access configuration settings
db_host = config['database']['host']
db_port = config['database']['port']
model_type = config['model']['type']
n_estimators = config['model']['parameters']['n_estimators']

