import json
import copy
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from src.utils.utils import check_save_dir, remove_files_with_prefix




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







if __name__ == "__main__":
    # Load experiment config
    with open("./src/config.json", 'r') as f:
        config = json.load(f)

    # Define save directory & tensorboard writer
    save_dir = f"./logs/{config['experiment_name']}"
    check_save_dir(save_dir)
    shutil.copy("./src/config.json", f"logs/{config['experiment_name']}/")
    writer = SummaryWriter(log_dir=save_dir)
    writer.flush()

    # Load data
    df = pd.read_csv(f"./data/{config['data_file']}", index_col=0)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)                   # shuffle rows    
    df_train, df_val = train_test_split(df, test_size=config['validation_pct'])

    X_train = torch.tensor(df_train.drop(columns=['SoC']).values, dtype=torch.float32)
    y_train = torch.tensor(df_train['SoC'].values, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(df_val.drop(columns=['SoC']).values, dtype=torch.float32)
    y_val = torch.tensor(df_val['SoC'].values, dtype=torch.float32).reshape(-1, 1)

    # Initialise model
    n_inputs = len(df_train.drop(columns=['SoC']).columns)
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
                    # Save best model at log frequenty
                    if loss_eval < best_loss:
                        best_loss = loss_eval
                        best_weights = copy.deepcopy(model.state_dict())
                        remove_files_with_prefix(dir=save_dir, prefix="best_model")
                        torch.save(best_weights, save_dir + f'/best_model_{log_idx}.pt')
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