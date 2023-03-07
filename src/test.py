import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

import torch

from src.main import Net


def read_mat_file(file):
    df = pd.DataFrame()
    columns = ['TimeStamp', 'Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time', 'Chamber_Temp_degC']
    for col in columns:
       df[col] = pd.Series(file['meas'][0][0][col].flatten())
    return df


def load_model(experiment_path):
    # extract model from experiment
    torch_models = [file for file in os.listdir(experiment_path) if file.endswith('.pt')]
    if len(torch_models) < 1: 
        print('No models found in experiment. Aborting ...')
        sys.exit()
    elif len(torch_models) > 1:
        print('Multiple models in experiment. Aborting ...')
        sys.exit()
    else:
        model_path = experiment_path + '/' + torch_models[0]

    # loading model
    model = Net(n_inputs=4) # TODO: make agnostic
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def plot_estimates(y_tests, y_preds, times, MAEs, labels, temp):
    fig, axs = plt.subplots(len(y_tests), 2, figsize=(16, 8), sharex=True)
    fig.suptitle(f'Temperature ({temp}), MAE: {round(np.mean(MAEs)*100, 3)}%', fontsize="x-large")
    for y_test, y_pred, time, MAE, label, ax in zip(y_tests, y_preds, times, MAEs, labels, axs):
        ax[0].plot(time, y_test, label="Measured")
        ax[0].plot(time, y_pred, label="Predicted")
        ax[0].set_ylabel("SOC \n(%)")
        ax[0].legend()

        ax[1].plot(time, (y_test-y_pred)/y_test*100)
        ax[1].set_ylabel("SOC \nerror \n(%)")
    
        ax[0].text(-0.15, 0.5, f"{label} \nMAE: {round(MAE*100, 2)}%", va='center', ha='right', rotation='horizontal', transform=ax[0].transAxes, fontweight='bold')

    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    plt.show()



if __name__ == "__main__":
    
    # Load model
    EXPERIMENT_TOTEST = "panasonic-initial-0.0.1"
    model = load_model(experiment_path=f"./logs/{EXPERIMENT_TOTEST}")
    
    #TODO: Same as in preprocess and feature selection (make seperate stage and save preprocessed drive cycle files per experiment)
    
    # Panasonic Data
    panasonic = "./data/Panasonic/Panasonic 18650PF Data"
    all_temperatures = ["0degC", "-10degC", "-20degC", "10degC", "25degC"]
    all_test_names = ["HWFET", "LA92", "UDDS", "US06", "Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN", "HWFTa", "HWFTb"]

    # dropped files are just concats of previous measurements
    dropped = [
        '06-01-17_10.36 0degC_LA92_NN_Pan18650PF.mat',
        '03-27-17_09.06 10degC_US06_HWFET_UDDS_LA92_NN_Pan18650PF.mat',
        '06-07-17_08.39 n10degC_US06_HWFET_UDDS_LA92_Pan18650PF.mat',
        '06-23-17_23.35 n20degC_HWFET_UDDS_LA92_NN_Pan18650PF.mat',
    ]

    # Test all Drive cycles
    for temp in all_temperatures:
        panasonic_dir = panasonic + f"/{temp}" + "/Drive cycles"
        panasonic_files = [f for f in os.listdir(panasonic_dir) if f not in dropped]
        labels = [label for file in panasonic_files for label in all_test_names if label in file]

        temp_MAEs, temp_y_tests, temp_y_preds, times = [], [], [], []
        for (file, label) in zip(panasonic_files, labels):           
            df_file = read_mat_file(sio.loadmat(panasonic_dir + "/" + file))
            
            # calculate SoC given a Panasonic 18650PF cell with a maximum capacity of 2.9Ah
            # Assumption: All test were started at an initial 100% SoC
            df_file["SoC"] = (2.9 + df_file['Ah'])/2.9
            
            # Moving Average (window of 400 timesteps - see paper)
            df_file["Voltage_MA400"] = df_file["Voltage"].rolling(400,min_periods=1).mean()
            df_file["Current_MA400"] = df_file["Current"].rolling(400,min_periods=1).mean()

            # Feature selection: remove obsolete columns
            time_from_start = np.array(df_file['Time'] - df_file['Time'][0])
            df_file = df_file.drop(['TimeStamp', 'Time', 'Current', 'Ah', 'Wh', 'Power', 'Chamber_Temp_degC'], axis=1)

            # Evaluate performance against model
            X_test = torch.tensor(df_file.drop(columns=['SoC']).values, dtype=torch.float32)
            y_test = torch.tensor(df_file['SoC'].values, dtype=torch.float32).reshape(-1, 1)
            
            y_pred = model(X_test)

            # Calculate performance
            MAE = torch.mean(torch.abs(y_test - y_pred))
            temp_MAEs += [float(MAE)]
            temp_y_tests += [y_test.cpu().detach().numpy()]
            temp_y_preds += [y_pred.cpu().detach().numpy()]
            times += [time_from_start]

        print(f'MAE ({temp}): \t{round(np.mean(temp_MAEs)*100, 3)}%')

        # Plot estimates
        plot_estimates(temp_y_tests, temp_y_preds, times, temp_MAEs, labels, temp)
        