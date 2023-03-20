import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

import shap
import torch
from tqdm import tqdm

from src.models import Net, DynamicNet


def read_mat_file(file):
    df = pd.DataFrame()
    columns = ['TimeStamp', 'Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time', 'Chamber_Temp_degC']
    for col in columns:
       df[col] = pd.Series(file['meas'][0][0][col].flatten())
    return df


def symmetric_derivative(y, t):
    """
    Calculates the symmetric difference quotient.
    """
    y, t = np.array(y), np.array(t)
    
    delta_2t = np.diff(t)[1:] + np.diff(t)[:-1]
    forward_step = y[2:]
    backward_step = y[:-2]
    
    return  np.concatenate(([np.nan], (forward_step - backward_step)/delta_2t, [np.nan]))


def load_model(experiment_path, n_inputs):
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
    if config["model"] == "Baseline":
        model = Net(n_inputs=n_inputs)
    if config["model"] == "DynamicDNN":    
        with open(experiment_path + "/params.json", "r") as f:
            params = json.load(f)
        model = DynamicNet(n_inputs=n_inputs, layer_sizes=params["model_layer_sizes"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def plot_estimates(y_tests, y_preds, times, MAEs, labels, temp, save_path=None):
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
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"{save_path}/mae_{temp}.png")
    plt.close()



def test_model(model, save_path):

    # Test all Drive cycles
    MAE_temp = {}
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
            
            # Moving Averages
            df_file["Voltage_MA1000"] = df_file["Voltage"].rolling(1000,min_periods=1).mean()
            df_file["Current_MA1000"] = df_file["Current"].rolling(1000,min_periods=1).mean()
            df_file["Voltage_MA400"] = df_file["Voltage"].rolling(400,min_periods=1).mean()
            df_file["Current_MA400"] = df_file["Current"].rolling(400,min_periods=1).mean()
            df_file["Voltage_MA200"] = df_file["Voltage"].rolling(200,min_periods=1).mean()
            df_file["Current_MA200"] = df_file["Current"].rolling(200,min_periods=1).mean()
            df_file["Voltage_MA100"] = df_file["Voltage"].rolling(100,min_periods=1).mean()
            df_file["Current_MA100"] = df_file["Current"].rolling(100,min_periods=1).mean()
            df_file["Voltage_MA50"] = df_file["Voltage"].rolling(50,min_periods=1).mean()
            df_file["Current_MA50"] = df_file["Current"].rolling(50,min_periods=1).mean()
            df_file["Voltage_MA10"] = df_file["Voltage"].rolling(10,min_periods=1).mean()
            df_file["Current_MA10"] = df_file["Current"].rolling(10,min_periods=1).mean()

            # Power
            df_file["Power"] = df_file["Voltage"]*df_file["Current"]

            # Derivatives
            # df_file["Voltage_grad"] = symmetric_derivative(df_file["Voltage"], df_file["Time"])
            # df_file["Current_grad"] = symmetric_derivative(df_file["Current"], df_file["Time"])
            # df_file["Battery_Temp_grad"] = symmetric_derivative(df_file["Battery_Temp_degC"], df_file["Time"])
            # df_file["Power_grad"] = symmetric_derivative(df_file["Power"], df_file["Time"])
            
            # Feature selection
            df_file = df_file.drop(['Chamber_Temp_degC'], axis=1)
            df_file.drop(df_file.tail(1).index, inplace=True)
            df_file.drop(df_file.head(1).index, inplace=True)
            time_from_start = np.array(df_file['Time']) - df_file['Time'].values[0]
            df_file = df_file[features + target]

            # Normalization
            with open(f"./data/{config['norm_basis']}", 'r') as f:
                norm_basis = json.load(f)                    
            for feature in norm_basis.keys():
                minmax = norm_basis[feature]
                df_file[feature] = (df_file[feature] - minmax[0])/(minmax[1] - minmax[0])           
            
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

        MAE_temp[temp] = round(np.mean(temp_MAEs)*100, 3)
    
        # Plot estimates
        plot_estimates(temp_y_tests, temp_y_preds, times, temp_MAEs, labels, temp, save_path=save_path)

    return MAE_temp
    

def analyse_feature_importance(model, save_path=None):
    # Feature importance analysis
    df = pd.read_csv(f"./data/{config['data_file']}", index_col=0)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X_total = torch.tensor(df.drop(columns=['SoC']).values, dtype=torchmodel.float32)
    
    X_sample = X_total[np.random.choice(X_total.shape[0], 1000, replace=False)]
    explainer = shap.DeepExplainer(model, X_sample)              
    shap_values = explainer.shap_values(X_sample)

    features = np.array(df.drop(columns=['SoC']).columns)
    df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
        "stdev_abs_shap": np.std(np.abs(shap_values), axis=0), 
    }, index=features)
    df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    print(df)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_size=[10,8], show=False)
    plt.title("Feature Importance") 
    plt.savefig(f"{save_path}/feature_importance.png")
    # plt.show()
    plt.close()



if __name__ == "__main__":
    
    # Load model
    EXPERIMENT_TOTEST = "panasonic-hyper-0.0.5"
    
    features = ['Voltage', 'Current', 'Power', 'Battery_Temp_degC', 'Voltage_MA1000', 'Current_MA1000', 'Voltage_MA400', 'Current_MA400', 'Voltage_MA200', 'Current_MA200', 'Voltage_MA100', 'Current_MA100', 'Voltage_MA50', 'Current_MA50', 'Voltage_MA10', 'Current_MA10']
    # , 'Voltage_grad', 'Current_grad', 'Battery_Temp_grad', 'Power_grad'
    target = ['SoC']
    with open(f"./logs/{EXPERIMENT_TOTEST}/config.json", 'r') as f:
        config = json.load(f)
        
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
    
    
    if config["hyperparameter_tuning"] == False:
    
        experiment_path = f"./logs/{EXPERIMENT_TOTEST}"
        model = load_model(experiment_path=experiment_path, n_inputs=len(features))
        
        # Test the model
        MAEs = test_model(model=model, save_path=experiment_path)
        for temp, mae in MAEs.items():
            print(f'MAE ({temp}): \t{mae}%')
    
        # Feature importance analysis
        analyse_feature_importance(model, save_path=experiment_path)
        
    elif config["hyperparameter_tuning"] == True:
        
        test_results = []
        hyper_folders = glob.glob(f"./logs/{EXPERIMENT_TOTEST}/train_*")[0]
        for trial_folder in tqdm(glob.glob(f"{hyper_folders}/trial_*")):
            
            model = load_model(experiment_path=trial_folder, n_inputs=len(features))
            
            # Test the model
            MAEs = test_model(model=model, save_path=trial_folder)
            
            # Feature importance analysis
            # analyse_feature_importance(model, save_path=trial_folder)     # (takes a lot of time)
            
            MAEs["trial"] = os.path.basename(trial_folder)
            
            test_results += [pd.DataFrame([MAEs])]
            
        # combine test results
        total_df = pd.concat(test_results)
        total_df['Average_MAE'] = total_df[all_temperatures].mean(axis=1)
        total_df.set_index("trial", inplace=True)

        total_df.to_csv(f"./logs/{EXPERIMENT_TOTEST}/test_results.csv")
        