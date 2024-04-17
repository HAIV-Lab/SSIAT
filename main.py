import json
import os
import argparse
from trainer import SSIAT_train

def main():
    args = parse_arguments()
    config = load_json(args.config)
    merged_config = merge_configs(args, config)
    SSIAT_train(merged_config)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    """
    Please specify the corresponding JSON file!!!!
    """
    parser.add_argument('--config', type=str, default='./exps/adapter_imageneta.json',
                        help='Json file of settings.')
    return parser.parse_args()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def merge_configs(args, config):
    merged_config = vars(args)  
    merged_config.update(config)  
    return merged_config

if __name__ == '__main__':
    main()