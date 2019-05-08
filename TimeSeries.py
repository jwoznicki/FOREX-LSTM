import numpy as np
import pandas as pd
import datetime
import os
import itertools
import torch
from torch.utils import data as torchdata

import ProcessData
from utils import Definitions

import warnings
warnings.simplefilter('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns; sns.set()
from pylab import rcParams

import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 18

class Preprocessor(ProcessData.Preprocessor):
    '''Class to manipulate time series data.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Split for train, validation, test using OOT samples
    def train_test_split(self, data):
        # Do different things if it's a numpy array vs dataframe
        if isinstance(data, pd.DataFrame):
            return super().train_test_split(data)
        else:
            return self._train_test_split_arr(data)
    
    # Split numpy array
    def _train_test_split_arr(self, data:np.ndarray):
        train_split = np.round(data.shape[0]*(1.0 - 2*self.test_size)).astype(int)
        val_split = np.round(data.shape[0]*(1.0 - self.test_size)).astype(int)

        data_train = data[0:train_split]
        data_val = data[train_split:val_split]
        data_test = data[val_split:]
        
        return data_train, data_val, data_test
    
    # Prep the data by transforming all columns to the proper type and encoding. Sets the date as the index
    def retype(self, data:pd.DataFrame):
        try:
            # Rename the date column for consistency
            if 'date' in data.columns:
                data['Date'] = data['date']
                data.drop(columns=['date'], inplace=True)
            else:
                pass

            # Set the date column to the index
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index(pd.DatetimeIndex(data['Date'], freq='infer'), inplace=True)
            else:
                data.set_index(pd.DatetimeIndex(data.index, freq='infer'), inplace=True)
            
        except:
            print('Unable to make the index a datetime. Please confirm date is available in the dataset.')
        
        # Retype normally
        return super().retype(data)

            
    # Impute missing data
    def impute(self, data:pd.DataFrame):
        for var in data.columns:
            # Get the variable type
            var_type = Definitions.column_types.get(var,'unknown')

            # Numeric - fill in missing numeric values with the previous time periods value. If no previous value, impute with the default
            if var_type == 'numeric':
                data[var] = data[var].ffill()
            # All other types, do nothing yet
            else:
                pass
            
        # finally, do what we do with the other types
        return super().impute(data)
    
    
    # Reshape data for time series
    def reshape(self, data:np.ndarray):
        total_time = data.shape[0]
        feature_size = data.shape[1]
        
        if total_time < self.time_steps:
            print('Warning - predicting with this data may result in inaccuracte predictions. Please provide at least '+str(self.time_steps)+' time steps in the past if possible.')
        
        # Reshape the data to B X N X C
        X_transformed = np.ndarray(shape=(total_time, self.time_steps, feature_size))

        # X is the current price and the prior n steps, Y is the next step
        for i in range(total_time):
            if i < self.time_steps:
                tmp_size = self.time_steps-(i+1)
                X_transformed[i,:,:] = np.concatenate((np.zeros(shape=(tmp_size,feature_size), dtype=float), data[0:(i+1),]))
            else:
                X_transformed[i,:,:] = data[((i+1)-self.time_steps):(i+1),:]
        
        return X_transformed
    
    # fit our data for transformation
    def fit(self, data:pd.DataFrame, features:[], targets:[], test_size:float, time_steps:int):
        self.time_steps = time_steps
        super().fit(data=data, features=features, targets=targets, test_size=test_size)
    
        
    # transform the data as we normally would, then reshape for LSTMs
    def transform(self, data:pd.DataFrame, train:bool=False):
        X, Y = super().transform(data, train)
        
        # Add target into our features
        X = X.values
        Y = Y.values
        X = np.hstack((Y, X))
        X = self.reshape(X)
                
        # Training vs Predicting
        if train:
            # Remove the first row of Y and last row of X to make X & Y the same shape
            X = X[:-1,:,:]
            Y = Y[1:,:]
            
            return X, Y
          
        # Predicting
        else:
            # We only care about X, since we don't know Y
            return X
        
        
    def dataloader(self, data:np.ndarray, batch_size:int=128):
        # Split out train, val, and test
        train, val, test = self.train_test_split(data)

        # Make them tensors
        train = torch.from_numpy(train).float()
        val = torch.from_numpy(val).float()
        test = torch.from_numpy(test).float()

        # Make tensor datasets and add them to the dataloader
        train_dataloader = torchdata.DataLoader(torchdata.TensorDataset(train), batch_size=batch_size, drop_last=False, pin_memory=False)
        val_dataloader = torchdata.DataLoader(torchdata.TensorDataset(val), batch_size=batch_size, drop_last=False, pin_memory=False)
        test_dataloader = torchdata.DataLoader(torchdata.TensorDataset(test), batch_size=batch_size, drop_last=False, pin_memory=False)

        return train_dataloader, val_dataloader, test_dataloader

            
# Class for Visualizing Time Series Data
class VizualizeTimeSeries:
    '''Class to visualize time series data'''
    
    def __init__(self, data:pd.DataFrame, name:str='', time_len:str='', var:str='', *args, **kwargs):
        self.data = data
        self.name = name
        self.title = Definitions.df_defs.get(name,'')
        self.time_len = time_len 
        self.var = var
        self.var_defn = Definitions.column_defs.get(var,'')
        self.var_units = Definitions.column_units.get(var,'')
        self.pct_format = '{x:,.1%}'
        self.dollar_format = '{x:,.2f}'
        self.count_format = '{x:,.0f}'
        self.numeric_format = '{x:,.1f}'
    
    
    # Create a plot for Visulaization
    def _create_plot(self, num_rows:int=1, num_cols:int=1):        
        fig = plt.figure(figsize=(24, 8))
        ax = fig.subplots(num_rows, num_cols, sharex=True)
        return fig, ax
    
    
    # save and output data, if necessary
    def save_show_plot(self, show_plots:bool, plot_name:str=''):
        if self.time_len == '':
            time_len = self.time_len
        else:
            time_len = self.time_len + '_'
        
        # Output the pot as a png
        plotPath = os.path.join('images', 'EDA', self.name, self.var, self.name + '_' + self.var + '_' + time_len + plot_name +'.png')
        
        # Create the directory if it does not exits
        if not os.path.exists(os.path.dirname(plotPath)):
            os.makedirs(os.path.dirname(plotPath))
        else:
            pass
            
        # save the plot
        plt.savefig(plotPath)
        
        # Output if requested
        if show_plots:
            plt.show()
        else:
            pass
        
        plt.close()

    def format_date_axis(self, ax):
        time_diff = (self.data.index.max() - self.data.index.min())
        ax.set_xlabel('Date')
        
        if time_diff.days >= (20*365):
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif time_diff.days >= (2*365):
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        elif time_diff.days >= (2*31):
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M %Y'))
        elif tiime_diff.days >= 15:
            ax.xaxis.set_major_locator(mdates.DayLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M %b %Y'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M %b %Y'))
        
        return ax
    
    # Plot the data over time
    def plot_line_series(self, show_plots:bool):
        if self.time_len == '':
            time_len = self.time_len
        else:
            time_len = ' (' + self.time_len + ')'
            
        fig, ax = self._create_plot()
        sns.lineplot(data=self.data, ax=ax)
        ax.set_title(self.title + ': ' + self.var_defn + ' over time'+time_len, ha='left', x=0, va='bottom')
        ax.set_ylabel(self.var + ' ' + self.var_units)
        ax = self.format_date_axis(ax)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(self.dollar_format))
        ax.axhline(y=0, color='#414141', linewidth=1.5, alpha=.5)
        ax.get_legend().remove()
        self.save_show_plot(show_plots=show_plots, plot_name='series')

    # Plot the distribution of the data
    def plot_distribution(self, show_plots:bool):
        if self.time_len == '':
            time_len = self.time_len
        else:
            time_len = ' (' + self.time_len + ')'
        
        fig, ax = self._create_plot()
        sns.distplot(self.data, ax=ax)
        ax.set_title(self.title + ': ' + self.var_defn+' distribution'+time_len, ha='left', x=0, va='bottom')
        ax.set_ylabel('Distribution')
        ax.set_xlabel(self.var + ' ' + self.var_units)
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(self.numeric_format))
        ax.axhline(y=0, color='#414141', linewidth=1.5, alpha=.5)
        self.save_show_plot(show_plots=show_plots, plot_name='distribution')

    # Plot the autocorrelation
    def plot_acf_(self, ax):
        if self.time_len == '':
            time_len = self.time_len
        else:
            time_len = ' (' + self.time_len + ')'
            
        tsaplots.plot_acf(self.data, ax=ax, lags=365)
        ax.set_title(self.title + ': ' + self.var_defn+' Autocorrelation and Partial Autocorrelation'+time_len, ha='left', x=0, va='bottom')
        ax.set_ylabel('Corr')

    # Plot the partial autocorrelation
    def plot_pacf_(self, ax):
        tsaplots.plot_pacf(self.data, ax=ax, lags=365)
        ax.set_title('')
        ax.set_ylabel('Corr')
        ax.set_xlabel('Lags')

    # Plot the autocorrelation and partial autocorrelation
    def plot_acf_pcf(self, show_plots:bool):
        fig, ax = self._create_plot(num_rows=2, num_cols=1)
        self.plot_acf_(ax[0])
        self.plot_pacf_(ax[1])
        self.save_show_plot(show_plots=show_plots, plot_name='acf_pcf')

    # Plot the seasonal decomposition
    def plot_decomposition(self, show_plots:bool):
        freq = 7 if self.time_len == 'daily' else 12
        fig, ax = self._create_plot(num_rows=4, num_cols=1)
        decomposition = sm.tsa.seasonal_decompose(self.data, model='additive', freq=freq)
        
        # Plot each piece
        decomposition.observed.plot(ax=ax[0], legend=False)
        ax[0].set_ylabel('Observed')
        decomposition.trend.plot(ax=ax[1], legend=False)
        ax[1].set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax[2], legend=False)
        ax[2].set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax[3], legend=False)
        ax[3].set_ylabel('Residual')
        
        
        if self.time_len == '':
            time_len = self.time_len
        else:
            time_len = ' (' + self.time_len + ')'
        
        # Change the big stuff
        ax[0].set_title(self.title + ': ' + self.var_defn+' Decomposition'+time_len, ha='left', x=0)
        ax[3] = self.format_date_axis(ax[3])
        self.save_show_plot(show_plots=show_plots, plot_name='decomposition')
        
    def plot_corr(self, show_plots:bool=False):
        fig, ax = self._create_plot()
        corr = self.data.corr()
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
        self.save_show_plot(show_plots=show_plots, plot_name='correlation')

    # Plot a numeric column
    def plot_numeric(self, show_plots:bool=False):
        # Plot the time series
        self.plot_line_series(show_plots)

        # Plot histogram of data
        self.plot_distribution(show_plots)

        # Add the auto and partial autocorrelations
        self.plot_acf_pcf(show_plots)

        # Add the seasonal decomposition
        self.plot_decomposition(show_plots)
        
    # plot a categorical column
    def plot_categorical(self, show_plots_bool=False):
        pass
    
    # plot a boolean column
    def plot_bool(self, show_plots_bool=False):
        pass
    
    
# Create charts to view the tiem
def visualize(data:pd.DataFrame, name:str, time_len:str, targets:[], show_plots:bool=False):
    # plot the correlation between the variables
    viz = VizualizeTimeSeries(data=data, name=name, time_len=time_len)
    viz.plot_corr(show_plots)
    
    # Plot the variables
    for var in targets:
        var_type = Definitions.column_types.get(var,'unknown')

        if var_type == 'numeric':
            viz = VizualizeTimeSeries(data=data[[var]], name=name, time_len=time_len, var=var)
            viz.plot_numeric(show_plots)
        elif var_type == 'categorical':
            viz = VizualizeTimeSeries(data=data[[var]], name=name, time_len=time_len, var=var)
            viz.plot_categorical(show_plots)
        elif var_type == 'bool':
            viz = VizualizeTimeSeries(data=data[[var]], name=name, time_len=time_len, var=var)
            viz.plot_bool(show_plots)
        else:
            pass
