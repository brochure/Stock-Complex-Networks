from core import Polygonal, Triangular, PartialTriangular, DcpsIdx, ROOTPATH, DROPLIST
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL, df_codes_and_title, DICT_STP
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
import sys, csv, time, requests, statsmodels, math
import pandas as pd
import numpy as np
from scipy import stats

'''
rdm_idx = np.random.randint(0, LENTRGL, size=10000)
rdm_pair = []
for idx in rdm_idx:
    rdm_pair.append(DcpsIdx(idx, LENTCKR))
    
df_rdm_pair = pd.DataFrame(rdm_pair)
df_rdm_pair.columns = ['i', 'j']
df_rdm_pair.to_csv(ROOTPATH + r'/Source/DF_RDM_PAIR_0612.csv', index=False)
'''

FILE_RDM_PAIR = ROOTPATH + r'/Source/DF_RDM_PAIR_0612.csv'
df_rdm_pair = pd.read_csv(FILE_RDM_PAIR)
FILE_RES_SAMPLE = ROOTPATH + r'/Source/RES_SAMPLE_0613.csv'
res_sample = pd.read_csv(FILE_RES_SAMPLE)
lst_rdm_pair = np.array(df_rdm_pair).tolist()
srs_res_sample = res_sample['res']

def ProcessCausality(pv1, pv2, threshold):
    if pv1 < threshold:
        if pv1 < pv2:
            return 1
        else:
            return -1
    elif pv2 < threshold:
        return -1
    return 0

def GCTestOnSglPair(S1, S2, max_lag=5):
    if len(S1) != len(S2):
        intsct = set(S1.index).intersection(set(S2.index))
        S1 = S1[intsct]
        S2 = S2[intsct]
    res_1 = grangercausalitytests(np.column_stack((S1, S2)), maxlag=max_lag, verbose=False)
    res_2 = grangercausalitytests(np.column_stack((S2, S1)), maxlag=max_lag, verbose=False)
    res_pair = GetGCTestResultPair(res_1, res_2)
    sigfct_threshold = 0.1
    if IsPairSeriesCross(res_pair):
        sigfct_threshold = 0.05
    for k in range(1, max_lag+1):
        idctr = ProcessCausality(res_1[k][0]['lrtest'][1], res_2[k][0]['lrtest'][1], sigfct_threshold)
        if idctr != 0:
            return idctr
    return 0

def GetGCTestResultPair(res_1, res_2):
    res = [None] * len(res_1)
    for i in range(len(res_1)):
        res[i] = (res_1[i+1][0]['lrtest'][1], res_2[i+1][0]['lrtest'][1])
    return res

def IsPairSeriesCross(GCTestResPair):
    flag_corss = False
    state = False
    if GCTestResPair[0][0] > GCTestResPair[0][1]:
        state = True
    for p in GCTestResPair[1:]:
        if p[0] > p[1]:
            if not state:
                flag_corss = True
        else:
            if state:
                flag_corss = True
    return flag_corss

def GetResFromRandomSample(rdm_pair):
    cnt = 0
    res = []
    for i, j in rdm_pair:
        inner_start = time.time()
        tmp = GCTestOnSglPair(DICT_STP[lst_tickers_stp[i]].log_return, DICT_STP[lst_tickers_stp[j]].log_return)
        print(tmp)
        res.append(tmp)
        print('cnt: ' + str(cnt) + ', time interval ' + str(i) + ': ' + str(time.time() - inner_start) + ';')
        cnt += 1
    return res
    