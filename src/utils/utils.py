import os
import sys
import shutil

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from ray import tune


def check_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(os.listdir(save_dir)) != 0:
        proceed = input('Experiment directory not empty, continue? [y/n]: ')
        if proceed != 'y':
            print('Aborted')
            sys.exit()
        print("Cleaning Experiment directory")
        delete_files_from_dir(save_dir)


def delete_files_from_dir(dir):
    for root, dirs, files in os.walk(dir, topdown=False):
        # Delete files
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {file_path}: {e}")
        
        # Delete directories
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Error deleting directory: {dir_path}: {e}")


def remove_files_with_prefix(dir, prefix):
    for filename in os.listdir(dir):
        if filename.startswith(prefix):
            os.remove(os.path.join(dir, filename))


def load_data(config, data_dir):
    df = pd.read_csv(f"{data_dir}/{config['data_file']}", index_col=0)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)                   # shuffle rows    
    df_train, df_val = train_test_split(df, test_size=config['validation_pct'])

    X_train = torch.tensor(df_train.drop(columns=['SoC']).values, dtype=torch.float32)
    y_train = torch.tensor(df_train['SoC'].values, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(df_val.drop(columns=['SoC']).values, dtype=torch.float32)
    y_val = torch.tensor(df_val['SoC'].values, dtype=torch.float32).reshape(-1, 1)
    
    return X_train, X_val, y_train, y_val


def get_hyper_parameter_config(hyper_dict, config):
    """
    Allowed types: 
        - loguniform --> info = [lower_bound, upper_bound]
        - TODO: extend allowed types
    """
    
    hyper_config = {}
    for param, spec in hyper_dict.items():
        if spec["type"] == "loguniform":
            hyper_config[param] = tune.loguniform(spec["info"][0], spec["info"][1])
        # TODO: extend this
        
    # extend with fixed config parameters
    for key, value in config.items():
        if key not in hyper_config.keys():
            hyper_config[key] = value
    
    return hyper_config
