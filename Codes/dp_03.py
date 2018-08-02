import sys, csv, time, requests, statsmodels, math
import pandas as pd
import numpy as np
from scipy import stats
import scipy.special
from contextlib import contextmanager
from datetime import datetime, timedelta
from dateutil.parser import parse
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
from core import SendEmail, ROOTPATH
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL, df_codes_and_title, DICT_STP

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 25)

FILE_EIO_2016 = ROOTPATH + '/Stock_DWCN/Source/lxl/EIO_2016.csv'
#####

# chosen_tickers = np.random.choice(lst_tickers_stp, 50, replace=False)

# establish source dataset X, Y
# 1st plausible sample proposition - fundamental, IEAs, corr -> coint (categorised)

# correlation
def SrcCorr(clsType=False, threshold=0.8):
    lst_trgl = [None] * LENTRGL
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        S = DICT_STP[lst_tickers_stp[i]].log_return
        for j in range(i+1, LENTCKR):
            if clsType:
                if S.corr(DICT_STP[lst_tickers_stp[j]].log_return) > threshold:
                    lst_trgl[cnt] = 1
                else:
                    lst_trgl[cnt] = 0
            else:
                lst_trgl[cnt] = S.corr(DICT_STP[lst_tickers_stp[j]].log_return)
            cnt += 1
        inner_end = time.time()
        print('Time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
    return lst_trgl

def SrcUnitRootTestRes():
    lst_trgl = [None] * LENTCKR
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        lst_log_return = DICT_STP[lst_tickers_stp[i]].log_return
        pvalue = adfuller(lst_log_return, regression='ct')[1]
        if pvalue < 0.01:
            lst_trgl[cnt] = 1
        elif pvalue < 0.05:
            lst_trgl[cnt] = 2
        else:
            lst_diff_1 = lst_log_return.diff(1)
            pvalue = adfuller(lst_diff_1[lst_diff_1.notna()], regression='ct')[1]
            if pvalue < 0.01:
                lst_trgl[cnt] = 3
            elif pvalue < 0.05:
                lst_trgl[cnt] = 4
            else:
                lst_diff_2 = lst_diff_1.diff(1)
                pvalue = adfuller(lst_diff_2[lst_diff_2.notna()], regression='ct')[1]
                if pvalue < 0.01:
                    lst_trgl[cnt] = 5
                elif pvalue < 0.05:
                    lst_trgl[cnt] = 6
                else:
                    lst_trgl[cnt] = 7
        inner_end = time.time()
        print('Flag: ' + str(lst_trgl[cnt]) + ', time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
        cnt += 1
    return lst_trgl
    
# cointegration
def SrcCointPval(clsType=False, threshold=0.01):
    lst_trgl = [None] * LENTRGL
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        S1 = DICT_STP[lst_tickers_stp[i]].log_return
        for j in range(i+1, LENTCKR):
            S2 = DICT_STP[lst_tickers_stp[j]].log_return
            if len(S1) != len(S2):
                intsct = set(S1.index).intersection(set(S2.index))
                S1 = S1[intsct]
                S2 = S2[intsct]

            _, pval, _ = coint(S1, S2)
            if clsType:
                if pval < threshold:
                    lst_trgl[cnt] = 1
                else:
                    lst_trgl[cnt] = 0
            else:
                lst_trgl[cnt] = pval
            cnt += 1
        inner_end = time.time()
        print('P-value: ' + str(pval) +'; Time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
    return lst_trgl

# IEAs
def SrcIEAs():
    lxl_matrix = np.matrix(np.genfromtxt(open(FILE_EIO_2016, 'rb'), delimiter=',', skip_header=2))
    lxl_industry_BEA_code_list = list(pd.read_csv(FILE_EIO_2016, nrows=0).columns)[:-2]
    lst_trgl = [None] * LENTRGL
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        idx1 = lxl_industry_BEA_code_list.index(df_codes_and_title.loc[lst_tickers_stp[i], 'BEA'])
        for j in range(i+1, LENTCKR):
            idx2 = lxl_industry_BEA_code_list.index(df_codes_and_title.loc[lst_tickers_stp[j], 'BEA'])
            if lxl_matrix[idx2, idx1] == 0:
                lst_trgl[cnt] = float('inf')
            elif lxl_matrix[idx1, idx2] == 0:
                lst_trgl[cnt] = float('-inf')
            else:
                lst_trgl[cnt] = np.log(lxl_matrix[idx1, idx2] / lxl_matrix[idx2, idx1])
            cnt += 1
        inner_end = time.time()
        print('Time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
    n_ceiling = math.ceil(max([x for x in lst_trgl if x != float('inf')])) + 1
    n_floor = math.floor(min([x for x in lst_trgl if x != float('-inf')])) - 1
    lst_trgl = [n_ceiling if x == float('inf') else x for x in lst_trgl]
    lst_trgl = [n_floor if x == float('-inf') else x for x in lst_trgl]
    return lst_trgl

def SrcFdmtls(df_fdmtl):
    LENITEM = len(df_fdmtl.columns)
    lst2d_trgl = [None] * LENITEM
    for i in range(LENITEM):
        lst2d_trgl[i] = [None] * LENTRGL
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        for j in range(i+1, LENTCKR):
            for k in range(LENITEM):
                lst2d_trgl[k][cnt] = df_fdmtl.iloc[i, k] / df_fdmtl.iloc[j, k]
            cnt += 1
        inner_end = time.time()
        print('Time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
    return lst2d_trgl

def ProcessCausality(pv1, pv2):
    if pv1 < 0.1:
        if pv1 < pv2:
            return 1
        else:
            return -1
    elif pv2 < 0.1:
        return -1
    return 0

def SrcGranger():
    lst_trgl = [0] * LENTRGL
    cnt = 0
    for i in range(LENTCKR):
        inner_start = time.time()
        S1 = DICT_STP[lst_tickers_stp[i]].log_return
        for j in range(i+1, LENTCKR):
            print('j: ' + str(j))
            S2 = DICT_STP[lst_tickers_stp[j]].log_return
            if len(S1) != len(S2):
                intsct = set(S1.index).intersection(set(S2.index))
                S1 = S1[intsct]
                S2 = S2[intsct]
            res_1 = grangercausalitytests(np.column_stack((S1, S2)), maxlag=5)
            res_2 = grangercausalitytests(np.column_stack((S2, S1)), maxlag=5)
            for k in range(1,6):
                idctr = ProcessCausality(res_1[k][0]['lrtest'][1], res_2[k][0]['lrtest'][1])
                if idctr != 0:
                    lst_trgl[cnt] = k*idctr
                    print(lst_trgl[cnt])
                    break
            cnt += 1
        inner_end = time.time()
        print('Time interval ' + str(i) + ': ' + str(inner_end - inner_start) + ';')
    return lst_trgl
                
df_source = pd.DataFrame()
df_source['corr'] = SrcCorr(False)
SendEmail('corr')
df_source['coint_pvalue'] = SrcCointPval(False)
SendEmail('coint_pvalue')
df_source['ieas'] = SrcIEAs()
SendEmail('ieas')
lst2d_fdmtl = SrcFdmtls(df_fdmtl)
for i in range(len(df_fdmtl.columns)):
    df_source[df_fdmtl.columns[i]] = lst2d_fdmtl[i]
df_source.to_csv(ROOTPATH + r'/Stock_DWCN/Source/DF_SOURCE_0530.csv')
SendEmail('to_csv')
