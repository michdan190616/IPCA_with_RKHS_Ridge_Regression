import numpy as np
import pandas as pd
import os

############## ---------------------------------------------------- ##############

def standardize(df):

    '''
    Standardize the data.

    Parameters:
    - df (matrix): Data.

    Returns:
    - matrix: Standardized data.
    '''

    return (df - df.mean()) / df.std()

############## ---------------------------------------------------- ##############

def fill_missing(df):

    '''
    Fill missing values with 0.

    Parameters:
    - df (matrix): Data.

    Returns:
    - matrix: Data with missing values filled with 0.
    '''

    return df.fillna(0)

############## ---------------------------------------------------- ##############

def download_clean_data(folder_path, start_date,ending_date, N):

    '''
    Download and clean the data.

    Parameters:
    - folder_path (string): Path to the folder containing the data.
    - start_date (int): Starting date.
    - ending_date (int): Ending date.
    - N (int): Number of stocks to keep.

    Returns:
    - list of matrices: List of matrices corresponding to characteristics for each stock.
    - list of arrays: List of returns for each time period for the stocks.
    '''

    files = os.listdir(folder_path)

    datasets = []

    for file in files:
        if file == '.DS_Store':
            continue
        if int(file[15:23]) < start_date:
            continue
        if int(file[15:23]) > ending_date:
            continue
        df = pd.read_csv(folder_path + '/' + file,  encoding='utf-8')
        df['name'] = file
        datasets.append(df)

    datasets.sort(key=lambda x: x['name'][0]) # sort by date

    macro = pd.read_csv('Data/macro_data_amit_goyal.csv', encoding='utf-8')
    macro = macro[(macro['yyyymm']>int(str(start_date)[:-2]))&(macro['yyyymm']<int(str(ending_date)[:-2]))]

    data = []
    ret = []

    for i,df in enumerate(datasets):
        
        df['mcap'] = df['SHROUT'] * df['prc']
        df.drop(['permno', 'DATE', 'Unnamed: 0', 'mve0', 'prc', 'SHROUT', 'sic2', 'name'], axis=1, inplace=True)
        df.dropna(thresh=60, axis=0, inplace=True)
        df = df[df['RET'] > -1]
        df = df.sort_values(by=['mcap'], ascending=False)
        df.drop(['mcap'], axis=1, inplace=True)
        df = df.head(N) # keep only N stocks based on market cap
        ret.append(df['RET']-macro['Rfree'].values[i])
        df = df.drop(['RET'], axis=1)
        df = standardize(df)
        df = fill_missing(df)
        data.append(df)

    return data, ret