'''
quandl.ApiConfig.api_key = 'so6AYpEpawDLVcSE-MkC'
data_s = quandl.get_table('WIKI/PRICES', date='2009-1-28', ticker='AAPL')
data_f = quandl.get_table('SHARADAR/SF1', calendardate='2016-12-31', ticker='AAPL')
df_train = pd.DataFrame(lst_file_etr[1:],columns=lst_file_train[0])
df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])
scipy.special.binom(2, 1)
'''

def GetCorrGraphByTickers(ticker_list, threshold=0.8):
    n = len(ticker_list)
    G = nx.Graph()
    G.add_nodes_from(ticker_list)
    for i in range(n):
        print(i)
        T1 = ticker_list[i]
        S1 = DICT_STP[T1].log_return
        for j in range(i+1, n):
            T2 = ticker_list[j]
            S2 = DICT_STP[T2].log_return
            wt = S1.corr(S2)
            if abs(wt) > threshold:
                G.add_edge(T1, T2, weight=wt)
    return G

# G = GetCorrGraphByTickers(chosen_tickers)
# nx.draw_random(G)
# nx.draw_shell(G)

# test simple cointegrate btwn chosen stocks
def FindCointegratedPairs(ticker_list, threshold=0.05):
    n = len(ticker_list)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    for i in range(n):
        T1 = ticker_list[i]
        S1 = DICT_STP[T1].log_return
        for j in range(i+1, n):
            T2 = ticker_list[j]
            S2 = DICT_STP[T2].log_return
            if len(S1) != len(S2):
                intsct = set(S1.index).intersection(set(S2.index))
                S1 = S1[intsct]
                S2 = S2[intsct]
            score_matrix[i, j], pvalue, _ = coint(S1, S2)
            pvalue_matrix[i, j] = pvalue
            if pvalue < threshold:
                pairs.append((T1, T2))
    return score_matrix, pvalue_matrix, pairs

# score_matrix, pvalue_matrix, pairs = FindCointegratedPairs(chosen_tickers)