import os
import json
import copy
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from src.utils.utils import *

from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from functools import partial



class Net(nn.Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 16)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(16, 32)
        self.act2 = nn.ReLU()
        # third hidden layer
        self.hidden3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        # output layer
        self.hidden4 = nn.Linear(16, 1)
        self.act4 = nn.Sigmoid()
 
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X


    

def train(config, save_dir=None, data_dir=None, hyper_enabled=False):
    # Custom save_dir for during hyperparameter tuning
    if hyper_enabled:
        save_dir = session.get_trial_dir()
    
    #Initialize tensorboard writer
    writer = SummaryWriter(log_dir=save_dir)
    writer.flush()

    # Load data
    X_train, X_val, y_train, y_val = load_data(config, data_dir=data_dir)

    # Initialise model
    n_inputs = np.shape(X_train)[1]
    model = Net(n_inputs=n_inputs)
    summary(model=model)
    
    # Loss function and Optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Hold best model
    best_loss = np.inf
    best_weights = None
    history_loss_train, history_loss_eval = [], []

    log_idx = 0
    batch_start = torch.arange(0, len(X_train), config['batch_size'])
    num_batches_per_epoch = len(batch_start)
    for epoch in range(config['epochs']):
        model.train()
        with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i, start in enumerate(bar):
                # Select batch
                X_batch = X_train[start:start+config['batch_size']]
                y_batch = y_train[start:start+config['batch_size']]
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
                # Set progress & tensorboard logs
                bar.set_postfix(mse=float(loss))
                writer.add_scalar("Loss_step/train", float(loss), epoch * num_batches_per_epoch + i)
        
                if i % int(num_batches_per_epoch // config['log_freq']) == 0 and i != 0:
                    # Evaluate model
                    model.eval()
                    y_pred = model(X_val)
                    loss_eval = float(loss_fn(y_pred, y_val))
                    # Log evaluation result
                    writer.add_scalar("Loss/train", float(loss), log_idx)
                    writer.add_scalar("Loss/eval", float(loss_eval), log_idx)
                    history_loss_train.append(float(loss))
                    history_loss_eval.append(float(loss_eval))
                    log_idx += 1
                    model.train()  
                    # Save best model at log frequency
                    if loss_eval < best_loss:
                        best_loss = loss_eval
                        best_weights = copy.deepcopy(model.state_dict())
                        remove_files_with_prefix(dir=save_dir, prefix="best_model")
                        torch.save(best_weights, save_dir + f'/best_model_{log_idx}.pt')
                    # Report to tune at log frequency
                    if hyper_enabled:
                        tune.report(loss=float(loss), loss_eval=float(loss_eval))
                
    writer.flush()
    writer.close()
    
    # Plot learning curves
    plt.plot(history_loss_train, label="loss_train")
    plt.plot(history_loss_eval, label="loss_eval")
    plt.title("Learning curves")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig(f"{save_dir}/learning_curves.png")
    





if __name__ == "__main__":    
    # Load experiment config
    data_dir = os.path.abspath("./data")
    with open("./src/config.json", 'r') as f:
        config = json.load(f)
    hyper_config = None
    if "hyperparameter_tuning" in config.keys():
        if config["hyperparameter_tuning"] == True:
            hyper_config = get_hyper_parameter_config(config["hyperparemeters"], config)
    
    # Define save directory & 
    save_dir = os.path.abspath(f"./logs/{config['experiment_name']}")
    check_save_dir(save_dir)
    shutil.copy("./src/config.json", f"logs/{config['experiment_name']}/")
    
    # Start training 
    if hyper_config is None:
        train(save_dir=save_dir, data_dir=data_dir)
    elif hyper_config is not None:
        # scheduler = FIFOScheduler()
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=config["epochs"],                              # max epochs
            grace_period=config["hyperparameter_min_epochs"],    # min epochs
            reduction_factor=2)                                  # punish severity
        reporter = CLIReporter(
            parameter_columns=[param for param in config["hyperparemeters"].keys()],
            metric_columns=["loss", "loss_eval"])
        
        def trial_dir_creator(trial):
            return f"trial_{trial.trial_id}"
        
        result = tune.run(
            partial(train, save_dir=save_dir, data_dir=data_dir, hyper_enabled=True),
            config=hyper_config,
            num_samples=hyper_config["hyperparameter_samples"],
            local_dir=save_dir,
            scheduler=scheduler,
            progress_reporter=reporter,
            trial_dirname_creator=trial_dir_creator)
