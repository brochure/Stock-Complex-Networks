# preprocess_stocks
import sys, csv, time, requests, statsmodels, math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core import Polygonal, Triangular, ROOTPATH, DROPLIST

FILE_PRICES_ENTIRE = ROOTPATH + '/Stock_DWCN/Source/WIKI_PRICES_ENTIRE.csv'
FILE_CODES_TITLE = ROOTPATH + '/Stock_DWCN/Source/industry_code/STOCK_CODES_TITLE.csv'

def AppendLogReturn(df_price, drop_first=False):
    df_apd = df_price.copy()
    df_apd['log_return'] = np.log(df_price.close / df_price.close.shift())
    if drop_first:
        return df_apd.drop(df_apd.index[0])
    return df_apd

def GetStockByTicker(ticker_ambo, chunk_size=1000):
    iter_file = pd.read_csv(FILE_PRICES_ENTIRE, iterator=True, chunksize=chunk_size)
    return pd.concat([chunk[chunk['ticker'].apply(lambda x: x == ticker_ambo)] for chunk in iter_file])

def GetDictStockOfStpYear(stp_year, threshold=200, drop_list = []):
    DICT_STP = {}
    for tckr, group in df_stock_grouped:
        if tckr in dict_codes_and_title.keys() and (tckr not in drop_list):
            stp_g = group[(group.index >= (datetime(stp_year, 1, 1)-timedelta(days=1))) & (group.index <= datetime(stp_year, 12, 31))]
            if len(stp_g) > threshold:
                DICT_STP[tckr] = AppendLogReturn(stp_g, True)
    return DICT_STP

def getIndustryCodeByStockCode(stock_code_list, code_type='NAICS'):
    return list(df_codes_and_title.loc[stock_code_list, code_type])

iter_file = pd.read_csv(FILE_PRICES_ENTIRE, iterator=True, chunksize=1000)
df_stock_entire = pd.concat([chunk for chunk in iter_file])
# group by ticker
df_stock_entire.date = pd.to_datetime(df_stock_entire['date'], dayfirst=False)
df_stock_entire = df_stock_entire[['ticker', 'date', 'close', 'volume']].set_index('date')
df_stock_grouped = df_stock_entire.groupby('ticker')
del df_stock_entire
df_codes_and_title = pd.read_csv(FILE_CODES_TITLE).set_index('Ticker')
dict_codes_and_title = df_codes_and_title.T.to_dict()

DICT_STP = GetDictStockOfStpYear(2016, 252, DROPLIST)

lst_tickers_stp = sorted(DICT_STP.keys())
LENTCKR = len(lst_tickers_stp)
LENTRGL = Triangular(len(lst_tickers_stp)-1)
