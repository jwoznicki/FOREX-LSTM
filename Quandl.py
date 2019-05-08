import quandl
import numpy as np
import pandas as pd
from urllib import request

df_dict = {}

# Pull in our currency pairs per USD
def load_forex_data():
    try:
        quandl.ApiConfig.api_key = open("config\quandl_api_key.txt", "r").readlines()[0]
    except:
        pass
    
    # Dictionary to keep data stuff and quandl code
    quandl_codes = {
        'AUS':'BOE/XUDLADD',
        'CAD':'BOE/XUDLCDD',
        'JPY':'BOE/XUDLJYD',
        'GBP':'BOE/XUDLGBD',
        'ZAR':'BOE/XUDLZRD',
        'CHF':'BOE/XUDLSFD',
        'EUR':'BOE/XUDLERD',
    }
    
    # Load each dataframe and store it in a dictionary
    forex_df_dict = {}
    
    # Load the data
    for curr, code in quandl_codes.items():
        df = quandl.get(code)
        forex_df_dict[curr] = df
        
    df_dict.update(forex_df_dict)
    
    return df_dict