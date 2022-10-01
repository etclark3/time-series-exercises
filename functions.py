import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import DateFormatter
import seaborn as sns


# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt

def evaluate(target_var):
    '''
    This function will take the actual values of the target_var from validate, and the predicted values in yhat_df,
    and compute the rmse, rounding to 0 decimal places. It will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse



def plot_and_eval(train, validate, test, yhat_df, target_var):
    '''
    This function takes in the target variable(string), and returns a plot of the values of train for that variable, 
    validate, and the predicted values from yhat_df. It will also label the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()
    
    
    
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def plot(train, validate, test, yhat_df, target_var):
    '''Plot all of train, validate, and test target variable values'''
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], label='train')
    plt.plot(validate[target_var], label='validate')
    plt.plot(test[target_var], label='test')
    plt.plot(yhat_df[target_var], alpha=.5, label='projection')
    plt.title(target_var)
    plt.legend()
    plt.show()
    
def get_items(url='https://python.zgulde.net/api/v1/items?page=1'):
    '''
    function to retrieve item data from website.
    accepts an url, returns a dataframe with items information
    '''
    max_page = requests.get(url).json()['payload']['max_page'] + 1
    for i in range(1,max_page):
        url = url[:-1] + str(i)
        if i == 1:
            output = pd.DataFrame(requests.get(url).json()['payload']['items'])
        else:
            output = pd.concat([output, pd.DataFrame(requests.get(url).json()['payload']['items'])], 
                               ignore_index=True)
    return output




def get_stores(url='https://python.zgulde.net/api/v1/stores'):
    '''
    function to retrieve store data from website.
    accepts an url, returns a dataframe with store information
    '''
    return pd.DataFrame(requests.get(url).json()['payload']['stores'])


def get_sales(url = 'https://python.zgulde.net/api/v1/sales?page='):
    '''
    function to retrieve sales data from website.
    accepts an url, returns a dataframe with sales information
    '''
    max_page = requests.get(url+'1').json()['payload']['max_page'] + 1
    for i in range(1,max_page):
        new_url = url + str(i)
        if i == 1:
            output = pd.DataFrame(requests.get(new_url).json()['payload']['sales'])
        else:
            output = pd.concat([output, pd.DataFrame(requests.get(new_url).json()['payload']['sales'])], 
                               ignore_index=True)
    return output

def combine(sales, stores, items):
    '''
    function to combine three dataframes of store data
    accepts 3 dataframes and returns them joined into one dataframe
    
    '''
    combo = sales.merge(items, left_on='item', right_on='item_id')
    combo = combo.merge(stores, left_on='store', right_on='store_id')
    combo.drop(columns=['item', 'store'], inplace=True)

    return combo

def thestore():
    '''
    Retrieve locally cached data .csv file for the superstore dataset
    If no locally cached file is present then retrieve sales, items, 
    and store dataframes from the web, merge them, cache the file as a csv, 
    and return the merged dataframe
    '''
    # if file is available locally, read it
    filename = 'store_info.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        # retrieve and combine data
        sales = get_sales()
        stores = get_stores()
        items = get_items()
        
        df = combine(sales, stores, items)

        # Write that dataframe to disk for later. This cached file will prevent repeated large queries to the database server.
        df.to_csv(filename, index=False)
    
    return df


def getnprep_OPSD():
    OPSD = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    OPSD.Wind.fillna(119.1, inplace=True)
    OPSD.Solar.fillna(89.25, inplace=True)
    OPSD['Wind+Solar'].fillna(240.991, inplace=True)