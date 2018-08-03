## Combining Machine Learning and Complex Network to Study Stock Market ##
## Python Programme - 01 ##
## Function: read raw stock data and export cleaned, appended, ##
##  and filtered tables ##
import csv
# import quandl
import networkx as nx
import pandas as pd
import numpy as np
import scipy.special
#import urllib2
from bs4 import BeautifulSoup as bs
from contextlib import contextmanager
from core import SendEmail, ROOTPATH
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 10)
FILE_PRICES_ENTIRE = ROOTPATH + r'/Source/industry_code/WIKI_PRICES_ENTIRE.csv'
FILE_CROSSWALK = ROOTPATH + r'/Source/industry_code/SIC-to-NAICS-Crosswalk.csv'

def query_country(symbol):
    url = 'https://finance.yahoo.com/quote/' + symbol + '/profile?p=' + symbol
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    with snip_country(soup) as is_US:
        return is_US
    
@contextmanager
def snip_country(soup):
    try: yield 'United States' in str(soup.find_all('p')[0])
    except IndexError: yield -1
    except ValueError: yield -2
    except TypeError: yield -3

def getForeignTickers():
    lst_foreign = []
    for tckr in lst_tickers_stp:
        if not query_country(tckr):
            lst_foreign.append(tckr)
    return lst_foreign

def query_sic(symbol, readable = False):
    url = 'https://www.sec.gov/cgi-bin/browse-edgar?CIK=' + symbol.upper() + '&Find=Search&owner=exclude&action=getcompany'
    soup = bs(urllib2.urlopen(url).read(), 'lxml')
    with snip_sic(readable, soup) as sic: return sic

@contextmanager
def snip_sic(readable, soup):
    try:
        if readable:
            yield soup.p.text.split(' - ')[1].split('State location')[0]
        yield int(soup.find_all('a')[9].contents[0])
    except IndexError: yield -1
    except ValueError: yield -2
    except TypeError: yield -3

# read whole stock data
iter_file = pd.read_csv(FILE_PRICES_ENTIRE, iterator=True, chunksize=1000)
df_stock_entire = pd.concat([chunk for chunk in iter_file])
lst_ambo_entire = pd.unique(df_stock_entire['ticker'])

# read inductrial code crosswalk table
pd_crosswalk = pd.read_csv(FILE_CROSSWALK).set_index('SIC')
# reserve the first value for recurrent SCI codes
pd_crosswalk_trimed = pd_crosswalk.iloc[[i for i in range(len(pd_crosswalk)) if pd_crosswalk.index[i] not in pd_crosswalk.index[:i]]]
dict_crosswalk = pd_crosswalk_trimed.T.to_dict()

dict_ambo = dict()
for ambo in lst_ambo_entire:
    print(ambo)
    SIC_code = str(query_sic(str(ambo)))
    if SIC_code in dict_crosswalk:
        dict_ambo[ambo] = dict_crosswalk[SIC_code].values()[0]

df_dict = pd.DataFrame.from_dict(dict_ambo, orient='index')
df_dict.columns = ['ticker', 'NAICS']
df_dict.to_csv(ROOTPATH + r'/Source/industry_code/TICKER-to-NAICS-Dict.csv')
