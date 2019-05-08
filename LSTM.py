import numpy as np
import pandas as pd
import torch
import datetime
import torch
from torch.utils import data as torchdata
from torch import nn, optim

import TimeSeries
from utils import Definitions

import warnings
warnings.simplefilter('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns; sns.set()
from pylab import rcParams
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 18        
        
# Setup the LSTM
class LSTMNet(torch.nn.Module):
    '''Class to create an LSTM Model for Time Series Data'''
    
    def __init__(self, LSTMData, hidden_size:int=1, num_layers:int=1, dropout:float=0.0):
        super().__init__()
        self.LSTMData = LSTMData
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=(len(LSTMData.preprocessor.features)+len(LSTMData.preprocessor.targets))
                            , hidden_size=self.hidden_size
                            , num_layers=self.num_layers
                            , bias=True
                            , batch_first=True
                            , dropout=self.dropout
                            , bidirectional=False)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=(len(LSTMData.preprocessor.targets)))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        ht = torch.zeros([self.num_layers, x.size(0), self.hidden_size], device=self.device)
        ct = torch.zeros([self.num_layers, x.size(0), self.hidden_size], device=self.device)
        output, (ht, ct) = self.lstm.forward(x, (ht, ct))
        return self.linear.forward(output[:, -1])


# Store the information we need to train and predict using an LSTM
class LSTM():
    def __init__(self, data:pd.DataFrame, name:str='', time_len:str=''):
        self.data = data
        self.name = name
        self.time_len = time_len
    
    # Take a dataframe and make it into a TimeSeries DF, then return LSTM Data
    def load(self, features:[], targets:[], test_size:float, time_steps:int, show_plots:bool=False):        
        self.preprocessor = TimeSeries.Preprocessor()
        self.preprocessor.fit(data=self.data.copy(), features=features, targets=targets, test_size=test_size, time_steps=time_steps)
        
        # Visualize the targets
        retyped_data = self.preprocessor.retype(self.data)
        TimeSeries.visualize(data=retyped_data, name=self.name, time_len=self.time_len, targets=self.preprocessor.targets, show_plots=show_plots)   
        X,Y = self.preprocessor.transform(data=self.data.copy(), train=True)
        X_train_dataloader, X_val_dataloader, X_test_dataloader = self.preprocessor.dataloader(X)
        Y_train_dataloader, Y_val_dataloader, Y_test_dataloader = self.preprocessor.dataloader(Y)
        
        self.X_train_dataloader = X_train_dataloader
        self.X_val_dataloader = X_val_dataloader
        self.X_test_dataloader = X_test_dataloader
        self.Y_train_dataloader = Y_train_dataloader
        self.Y_val_dataloader = Y_val_dataloader
        self.Y_test_dataloader = Y_test_dataloader

        print('')
        print('Data prepared:')
        print('    Number of Features: {}'.format(X.shape[-1]))
        print('    Number of Outputs: {}'.format(Y.shape[-1]))
        print('    Lookback {} time steps'.format(time_steps))
        print('')
    
    
    # Create the model
    def create(self, hidden_size:int, num_layers:int, dropout:float):
        self.model = LSTMNet(self, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        
        print('Model:')
        print('    {}-layer LSTM with feed-forward linear output from {} hidden nodes.'.format(num_layers, hidden_size))
        if (num_layers) > 1 and (dropout > 0):
            print('    Utilizing dropout with probability {}'.format(dropout))
        print('')
    
    
    # Train the model
    def train(self, criterion_func:str, optimizer_func:str, num_epochs:int, lr:float=0.001, weight_decay:float=0.0):    
        ''' Function to train LSTM '''
        
        device = self.model.device
        self.model = self.model.to(device)
        
        
        loss_funcs = {
            'MSE':nn.MSELoss(),
            'MAE':nn.L1Loss(),
        }

        optimizers = {
            'Adam':optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'AdamW':optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0.0),
            #'SGD':optim.sgd()
        }
        
        self.criterion = loss_funcs.get('MSE').to(device)
        self.optimizer = optimizers.get('Adam')

        model_run_start_time = datetime.datetime.now()
        print('Training Model for {} epochs - start time: {}'.format(num_epochs, model_run_start_time))
        print('    Optimized using {}, learning rate: {}'.format(optimizer_func, self.optimizer.defaults['lr']))
        print('    Loss calculated with {}'.format(criterion_func))
        print('')

        # Save the losses
        train_losses = list()
        val_losses = list()
        
        for epoch in range(num_epochs):
            y_iter = iter(self.Y_train_dataloader)
            losses = list()
            for batch, X in enumerate(self.X_train_dataloader):
                X = X[0].to(device)
                
                Y = next(y_iter)
                Y = Y[0].to(device)

                # Put the model in training mode
                self.model.train()

                # Reset gradient
                self.optimizer.zero_grad()

                # Forward
                Y_hat = self.model.forward(X)
                loss = self.criterion(Y_hat, Y)

                # backpropagation to calculate gradients
                loss.backward()

                # Clip update the gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 6)
                self.optimizer.step()

                # add losses to logging, if loss is torch tensor use .item() to switch to numpy arrays (saves memory)
                losses.append(loss.item())

            train_losses.append(np.mean(losses))

            # set the model to evaluation mode
            self.model.eval()
            with torch.no_grad():
                y_val_iter = iter(self.Y_val_dataloader)
                losses = list()
                for batch, X_val in enumerate(self.X_val_dataloader):
                    X_val = X_val[0].to(device)

                    Y_val = next(y_val_iter)
                    Y_val = Y_val[0].to(device)

                    # make prediction on validation set
                    Y_hat_val = self.model.forward(X_val)

                    # get validation loss
                    loss = self.criterion(Y_hat_val, Y_val)

                    losses.append(loss.item())

            val_losses.append(np.mean(losses))
            
            if ((epoch % 250 == 0) or (epoch == (num_epochs-1))):
                print('Epoch {}, Train Loss: {}, Validation Loss: {}'.format(epoch, np.mean(train_losses), np.mean(val_losses)))  

        model_run_end_time = datetime.datetime.now()
        # Make our test prediction
        self.model.eval()
        with torch.no_grad():
            y_test_iter = iter(self.Y_test_dataloader)
            test_losses = list()
            for batch, X_test in enumerate(self.X_test_dataloader):
                X_test = X_test[0].to(device)

                Y_test = next(y_test_iter)
                Y_test = Y_test[0].to(device)

                # make prediction on validation set
                Y_hat_test = self.model.forward(X_test)

                # get validation loss
                test_loss = self.criterion(Y_hat_test, Y_test)

                test_losses.append(test_loss.item())
            
            test_loss = np.mean(test_losses)
            
        train_time = (model_run_end_time - model_run_start_time).total_seconds()
        baseline = self.naive_model()

        self.run_data = {'Train Time':train_time,
                         'training losses':train_losses,
                         'validation losses':val_losses,
                         'test loss':test_loss,
                         'baseline':baseline,
                        }

        
        print('Training completed at {}, taking: {} seconds. Test error: {} (vs baseline: {})'.format(model_run_end_time, train_time, test_loss, baseline))
        
        
    def predict(self, data:pd.DataFrame):
        data = data.copy()
        data = self.preprocessor.transform(data)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(data)
            
        return output
    
    
    def naive_model(self):
        Y_hat = self.X_test_dataloader.dataset.tensors[0][:,-1,0].squeeze()
        Y = self.Y_test_dataloader.dataset.tensors[0][:,0]
    
        # score
        loss = self.criterion(Y_hat, Y)
        
        return loss
        
        
    def visualize_train(self):
        fig = plt.figure(figsize=(24, 8))
        ax = fig.subplots(1, 1, sharex=True)
        
        lines = pd.DataFrame({'training loss':self.run_data['training losses'], 'validation loss':self.run_data['validation losses']}) 

        colors = ['blue','orange']
        for i, col in enumerate(lines):
            plt.plot(lines[col], marker='', color=colors[i], linewidth=1, alpha=0.9, label=col)
        ax.set_title(self.name+'/USD training and validation loss', ha='left', x=0, va='bottom')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()

        # save the plot
        plotPath = os.path.join('images', 'Output', self.name, self.name + '_TrainingPlot (' + self.time_len +').png')

        # Create the directory if it does not exits
        if not os.path.exists(os.path.dirname(plotPath)):
            os.makedirs(os.path.dirname(plotPath))
        else:
            pass

        # save the plot
        plt.savefig(plotPath)
        #plt.show()
        plt.close()

        
    def visualize_error(self):
        fig = plt.figure(figsize=(24, 8))
        ax = fig.subplots(1, 1, sharex=True)

        self.model.eval()
        with torch.no_grad():
            y_train_iter = iter(self.Y_train_dataloader)
            actual = list()
            train_pred = list()
            for batch, X_train in enumerate(self.X_train_dataloader):
                X_train = X_train[0].to(self.model.device)
                Y_train = next(y_train_iter)
                Y_train = Y_train[0].to(self.model.device)
                Y_hat_train = self.model.forward(X_train)
                actual.extend(Y_train.squeeze().tolist())
                train_pred.extend(Y_hat_train.squeeze().tolist())

            y_val_iter = iter(self.Y_val_dataloader)
            val_pred = list()
            for batch, X_val in enumerate(self.X_val_dataloader):
                X_val = X_val[0].to(self.model.device)
                Y_val = next(y_val_iter)
                Y_val = Y_val[0].to(self.model.device)
                Y_hat_val = self.model.forward(X_val)
                actual.extend(Y_val.squeeze().tolist())
                val_pred.extend(Y_hat_val.squeeze().tolist())

            y_test_iter = iter(self.Y_test_dataloader)
            test_pred = list()
            for batch, X_test in enumerate(self.X_test_dataloader):
                X_test = X_test[0].to(self.model.device)
                Y_test = next(y_test_iter)
                Y_test = Y_test[0].to(self.model.device)
                Y_hat_test = self.model.forward(X_test)
                actual.extend(Y_test.squeeze().tolist())
                test_pred.extend(Y_hat_test.squeeze().tolist())
        
        predictions = train_pred + val_pred + test_pred
        lines = pd.DataFrame(data={'Actual':actual, 'Predicted':predictions}, index=pd.DatetimeIndex(self.data.index[1:], freq='infer'))
        lines['Date_'] = lines.index
        train_end_date = lines.iloc[len(train_pred)]['Date_']
        val_end_date = lines.iloc[(len(train_pred)+len(val_pred))]['Date_']
        lines.drop(columns=(['Date_']), inplace=True)
        lines['Naive'] = lines[['Actual']].ffill().shift(periods=1).fillna(0)
        lines.to_csv('AUS.csv')
        lines.sort_index(inplace=True)
            
        colors = ['grey','blue','orange']
        for i, col in enumerate(lines[['Naive','Actual','Predicted']]):
            plt.plot(lines[col], marker='', color=colors[i], linewidth=1, alpha=0.9, label=col)
        ax.set_title(Definitions.df_defs.get(self.name,'') + ' actual vs predicted', ha='left', x=0, va='bottom')
        ax.set_ylabel(self.name+'/USD, scaled')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.axvline(x=train_end_date, color='#414141', linewidth=1, alpha=.5)
        ax.axvline(x=val_end_date, color='#414141', linewidth=1, alpha=.5)
        ax.legend()
        
        # save the plot
        plotPath = os.path.join('images', 'Output', self.name, self.name + '_Accuracy (' + self.time_len +').png')

        # Create the directory if it does not exits
        if not os.path.exists(os.path.dirname(plotPath)):
            os.makedirs(os.path.dirname(plotPath))
        else:
            pass

        # save the plot
        plt.savefig(plotPath)
        #plt.show()
        plt.close()

        