import sys, csv, time, requests, statsmodels, math
import numpy as np
import pandas as pd
from requests.auth import HTTPBasicAuth
from core import SendEmail, Polygonal, Triangular, ROOTPATH
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL

FILE_TICKER_FDMTL = ROOTPATH + r'/Source/DF_FDMTL_0609.csv'
df_fdmtl = pd.read_csv(FILE_TICKER_FDMTL).set_index('ticker')

AUTHKEYPAIR = [
    ['f554a1d0b28d48b5513b15a86c0ce7cd', '43086c305b973346c3737a8ca47c9559'],
    ['7a8a96eff593d81e683269ecfea2bd1c', 'b0c4dbb4873777e04b0d29c07185fd2a'],
    ['9247923eddc2ea1555da91085b79dc91', '5f8059a408f557a47617c489bca0d191'],
    ['8cc04599de40a006f3da592544f3dbdd', 'd280deb51b208ed6fed3fdd35244f350'],
    ['c077386eb61eac04b7392c806d7ce17d', '8b05e1f1c7765c2ba6acc0c7e74e821e'],
    ['785509724cbe24b540136a37f047b8fc', '6d9ff969a4ad8afb24d31323e498fb85'],
    ['aef598e89e7dba01e68bd09a7eafd2c4', '9a163849f0d18367c5600f27c2b448e6'],
    ['31340f6ae220cbd15caebe2121ef0bdd', '9408bb75fe9025cb97eda037d436f59d']
]

# Get fundamental data from web-api
def GetFundamentalFromWebByTicker(ticker, item, year, freq = 'daily', ppsize = 300, authdex = 0):
    url = 'https://api.intrinio.com/historical_data?identifier=' + ticker + '&item=' + item + '&start_date=' + str(year) + '-01-01' + '&end_date=' + str(year) + '-12-31' + '&frequency=' + freq + '&page_size='+ str(ppsize)
    result = requests.get(
        url,
        auth=HTTPBasicAuth(
            AUTHKEYPAIR[authdex][0],
            AUTHKEYPAIR[authdex][1]
        )
    )
    return result.json()['data']

# fundamental
def srcFdmtl(itemname, year, dexith=0):
    lst_fdmtl = [None] * LENTCKR
    for i in range(LENTCKR):
        print('i: ' + str(i) + ', dexith: ' + str(dexith))
        try:
            lst_fdmtl[i] = GetFundamentalFromWebByTicker(lst_tickers_stp[i], itemname, year, authdex = dexith)
        except:
            print(sys.exc_info()[0])
            dexith += 1
            if dexith < len(AUTHKEYPAIR):
                lst_fdmtl[i] = GetFundamentalFromWebByTicker(lst_tickers_stp[i], itemname, year, authdex = dexith)
                print('dexith in except: ' + str(dexith))
                continue
            else:
                break
    return lst_fdmtl

def genFdmtlAvg(lst_item):
    sis_avg = pd.Series(index=lst_tickers_stp)
    lst_nil = []
    for i in range(LENTCKR):
        acc = 0.0
        cnt = 0
        for j in range(len(lst_item[i])):
            if type(lst_item[i][j]['value']) == str:
                continue
            acc += lst_item[i][j]['value']
            cnt += 1
        if cnt == 0:
            lst_nil.append(i)
            continue
        sis_avg[lst_tickers_stp[i]] = acc / cnt
    return sis_avg, lst_nil

#df_fdmtl = pd.DataFrame(index=lst_tickers_stp)
lst_fdmtl = srcFdmtl('beta', 2016, 0)
sis_avg, _ = genFdmtlAvg(lst_fdmtl)
df_fdmtl['beta'] = np.nan
for tck in lst_tickers_stp:
    df_fdmtl.loc[tck, 'beta'] = sis_avg[tck]

df_fdmtl.index.name = 'ticker'
df_fdmtl.to_csv(ROOTPATH + r'/Source/DF_FDMTL_0609.csv')
