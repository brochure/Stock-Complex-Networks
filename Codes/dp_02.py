import csv
import networkx as nx
import pandas as pd
import numpy as np
# import scipy.special
# from contextlib import contextmanager
from datetime import datetime, timedelta
from core import SendEmail, ROOTPATH
# from dateutil.parser import parse
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 10)
FILE_TICKER_NAICS = ROOTPATH + r'/Stock_DWCN/Source/industry_code/TICKER-to-NAICS-Dict.csv'
FILE_NAICS_BEA = ROOTPATH + r'/Stock_DWCN/Source/industry_code/NAICS_Code_to_BEA_Code.csv'
FILE_BEA_Code_TITLE = ROOTPATH + r'/Stock_DWCN/Source/industry_code/BEA_Code_to_Title.csv'

df_ticker_NAICS = pd.read_csv(FILE_TICKER_NAICS).set_index('ticker')
dict_ticker_NAICS = df_ticker_NAICS.NAICS.to_dict()
del df_ticker_NAICS
# read crosswork of BEA codes and titles
df_NAICS_BEA = pd.read_csv(FILE_NAICS_BEA).set_index('NAICS_Code')
dict_NAICS_BEA = df_NAICS_BEA.BEA_Code.to_dict()
del df_NAICS_BEA
df_BEA_code_title = pd.read_csv(FILE_BEA_Code_TITLE).set_index('BEA_Code')
dict_BEA_code_title = df_BEA_code_title.BEA_Title.to_dict()
del df_BEA_code_title

def TickerCodesAndTitle(tickers):
    dict_codes_and_title = {}
    for tckr in tickers:
        naics_code = dict_ticker_NAICS[tckr]
        if naics_code in dict_NAICS_BEA.keys():
            bea_code = dict_NAICS_BEA[naics_code]
        else:
            continue
        if bea_code in dict_BEA_code_title.keys():
            bea_title = dict_BEA_code_title[bea_code]
        else:
            continue
        dict_codes_and_title[tckr] = dict([('NAICS', naics_code), ('BEA', bea_code), ('Title', bea_title)])
    return dict_codes_and_title
dict_codes_and_title = TickerCodesAndTitle(dict_ticker_NAICS.keys())
df_codes_and_title = pd.DataFrame(dict_codes_and_title).T
# df_codes_and_title.columns[0] = 'Ticker'
df_codes_and_title.to_csv(ROOTPATH + r'/Stock_DWCN/Source/industry_code/STOCK_CODES_TITLE.csv')
