import numpy as np
import pandas as pd
import datetime
import os
import itertools

from utils import Definitions

import warnings
warnings.simplefilter('ignore')


class Preprocessor():
    '''Class to manipulate time series data.'''
    
    def __init__(self, *args, **kwargs):
        self.mapping = {} # How are we mapping categorical columns
        self.scaler_params = {} # How are we scaling the data
        self.column_types = Definitions.column_types
    
    # Prep the data by transforming all columns to the proper type and encoding. Sets the date as the index
    def retype(self, data:pd.DataFrame):
        # Transform the data into all numeric
        for var in data.columns:
            var_type = self.column_types.get(var,'unknown')
            
            # For numeric columns, make them numeric
            if var_type == 'numeric':
                # set to numeric
                data[var] = pd.to_numeric(data[var])
            # For categorical, make them string
            elif var_type == 'categorical':
                # set to a string
                data[var] = data[var].astype(str)
            # For boolean, make them bool
            elif var_type == 'bool':
                # set to a string
                data[var] = data[var].astype(bool)
            # Don't touch undefined columns
            else:
                pass
            
        return data
    
    # Split dataframe
    def train_test_split(self, data:pd.DataFrame):
        train_split = np.round(data.shape[0]*(1.0 - 2*self.test_size)).astype(int)
        val_split = np.round(data.shape[0]*(1.0 - self.test_size)).astype(int)

        train = data.iloc[0:train_split]
        val = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]
        
        return train, val, test
    
    
    # Save parameters to scale the data
    def _fit_scaler(self, data:pd.DataFrame):
        for var in data.columns:
            # Get the variable type
            var_type = self.column_types.get(var,'unknown')
            
            # Numeric only - scale by subtracting the mean and dividing by standard deviation
            if var_type == 'numeric':
                mu = np.mean(data[var].dropna())
                sigma = np.std(data[var].dropna())
                self.scaler_params[var] = {'mu':mu, 'sigma':sigma}
            else:
                pass
    
        
    # Save encoding for data to change strings to floats and floats to strings
    def _fit_encoder(self, data:pd.DataFrame):
        # encode all variables
        for var in data.columns:
            # Get the variable type
            var_type = self.column_types.get(var,'unknown')

            # Categorical
            if var_type == 'categorical':
                # Pull the unique classes for categorical
                var_classes = data[var].fillna('other').unique().tolist()
                self.mapping[var] = var_classes
                
                # Copy the targets and features
                self._encoded_features = self.features
                self._encoded_targets = self.targets
                
                # Add new column names into features or targets...
                encoded_data = pd.get_dummies(data=data[[var]])
                if var in self._encoded_features:
                    self._encoded_features.remove(var)
                    self._encoded_features.extend(list(encoded_data.columns.values))
                if var in self._encoded_targets:
                    self._encoded_targets.remove(var)
                    self._encoded_targets.extend(list(encoded_data.columns.values))
            # Default case, do nothing
            else:
                pass

            
    # Impute missing data
    def impute(self, data:pd.DataFrame):
        for var in data.columns:
            # Get the variable type
            var_type = self.column_types.get(var,'unknown')
            
            # Categorical - Fill missing with 'other'
            if var_type == 'categorical':
                data[var].fillna('other', inplace=True)
            # Numeric - fill in missing numeric values with the median
            elif var_type == 'numeric':
                median = np.median(data[var].dropna())
                data[var].fillna(median, inplace=True)
            # Boolean - fill with false
            elif var_type == 'bool':
                data[var].fillna(False, inplace=True)
            # Catch-all. Do nothing
            else:
                pass

        return data
    
    
    # Scale the testing data from the saved parameters
    def scale(self, data:pd.DataFrame):
        for var in data.columns:
            # Get the variable type
            var_type = self.column_types.get(var,'unknown')
            
            # Numeric only - Fill in missing numeric values with the previous time periods value. If there is no previous time values, fill 0
            if var_type == 'numeric':
                params = self.scaler_params.get(var,{'mu':0,'sigma':1})
                sigma = params['sigma']                
                mu = params['mu']
                data[var] = data[var].apply(lambda x: (x-mu)/sigma)
            else:
                pass
        
        return data

    
    # encode testing data from saved column encodings
    def encode(self, data:pd.DataFrame):
        # encode all variables
        for var in data.columns:
            # Get the variable type
            var_type = self.column_types.get(var,'unknown')
            
            # Categorical
            if var_type == 'categorical':
                var_classes = self.mapping.get(var, [])
                # Map classes not in training data to "other"
                data[var] = data[var].apply(lambda x: x if x in var_classes else 'other')
                
                # one-hot encode
                encoded_data = pd.get_dummies(data=data[[var]])
                for col in encoded_data.columns.values:
                    self.column_types[col] = 'categorical'
                
                # Add in unmapped columns
                unmapped_cols = np.setdiff1d(var_classes,encoded_data.columns.values)
                for col in unmapped_cols:
                    encoded_data[col] = np.zeros(encoded_data.shape[0])
                
                # drop existing
                data.drop(columns=[var], axis=1, inplace=True)
                # Add in new
                data = pd.concat([data, encoded_data], axis=1)
                
                # Replace the features and targets
                self.features = self._encoded_features
                self.targets = self._encoded_targets
            # Boolean   
            elif var_type == 'bool':
                # change to 0/1, impute missing with 0
                data[var] = data[var]*1.0
            # Skip numeric, no encoding necessary
            elif var_type == 'numeric':
                pass
            # Default case, do nothing
            else:
                pass
        
        # Return the dataframe
        return data

    
    # Prepare the data for training
    def fit(self, data:pd.DataFrame, features:[], targets:[], test_size:float=0.2):
        self.test_size = test_size
        self.features = features
        self.targets = targets
        
        # Retype the data
        data = self.retype(data=data)
        
        # split into train, validation, and test. Fit the scaler and encoder parameters
        train, val, test = self.train_test_split(data)
        self._fit_scaler(train)
        self._fit_encoder(train)
        
    
    def transform(self, data:pd.DataFrame, train:bool=False):
        if train:
            # drop where the targets are missing
            data.dropna(subset=self.targets, inplace=True)
        
        # Retype the data
        data = self.retype(data)
        
        # Impute the data
        data = self.impute(data)
        
        # Encode the data
        data = self.encode(data)
        
        # scale the data
        data = self.scale(data)
        
        if train:
            # Split X's and Y's
            X = data[self.features]
            Y = data[self.targets]
            return X,Y
        else:
            return X
        
    
    def fit_transform(self, data:pd.DataFrame, test_size:float=0.2):
        self.test_size = test_size
        self.fit(data, self.test_size)
        return self.transform(data)