# import packages
import numpy as np
import pandas as pd
import datetime
import os

# Put the time series data into a special class for visualization and data preparation
import LSTM
from utils import Quandl

def main():
    df_dict, df_def = Quandl.load_forex_data()
    
    # Model parameters
    features = []#['SandP','TBill','BollingerLower','BollingerMiddle','BollingerUpper','RSI','MA10','EMA12','EMA26','MACD']
    targets = ['Value']
    
    # Data hyperparams
    time_steps = 365 # How far back are we looking?

    # LSTM hyperparams
    hidden_size = 5 # How many outputs from the LSTM layer itself, which will then pass thorugh a linear layer?
    num_layers = 1 # How many LSTM layers?
    dropout = 0.2 # If we have multiple layers, what rate are we applying dropout?

    # Training hyperparams
    num_epochs = 500
    loss = 'MSE' # Huber, KL-Div, MAE, MSE, 
    optimizer = 'Adam' # SGD, RMSProp

    # learning Rate
    # Momentum
    # weight decay
    # Regularization
    # Weight Initialization

    run_history = {}
    # For each data, load and manipulate the data, then train an lstm
    for i, df in enumerate(df_dict):
        print('Predicting '+df_def[df])
        print('')

        lstmModel = LSTM.LSTM(data=df_dict[df], features=features, targets=targets, df_name=df, title=df_def[df])
        lstmModel.load()
        lstmModel.create(hidden_size, num_layers, dropout)
        lstmModel.train(loss, optimizer, num_epochs)
        lstmModel.visualize_train()
        lstmModel.visualize_error()

        run_history[i] = lstmModel

        print('')
        print('')
        print('')
    
if __name__ == '__main__':
    main()