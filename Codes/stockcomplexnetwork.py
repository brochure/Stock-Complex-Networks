from __future__ import division
import logging
import logging.config
import sys, csv, time, requests, statsmodels, math
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
import statsmodels.api as sm
from scipy import stats
import scipy.special
from scipy.stats import describe
from scipy.linalg import circulant
from contextlib import contextmanager
from datetime import datetime, timedelta
from dateutil.parser import parse
import collections
import random
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import pylab
import powerlaw
import networkx as nx
from networkx.algorithms import community
import pandas as pd
import numpy as np
from core import SendEmail, ROOTPATH, sri_SP500_log_return, sri_SP500_close
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL, df_codes_and_title, DICT_STP, getIndustryCodeByStockCode

sns.set(color_codes=True)
data_dir = ROOTPATH + r'/Codes'

# function for generating PGD
def genPureDedirectedGraph(theta_1, theta_2, mat_1, mat_2):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for i in lst_tickers_stp:
        #print(i, end=' ')
        matidx_i = EIO_industry_BEA_code_list.index(
            df_codes_and_title.loc[i, 'BEA'])
        for j in lst_tickers_stp:
            if i != j:
                matidx_j = EIO_industry_BEA_code_list.index(
                    df_codes_and_title.loc[j, 'BEA'])
                a = mat_1[matidx_i, matidx_j]
                b = mat_2[matidx_i, matidx_j]
                if a > 0.0 and a > theta_1:
                    G.add_edge(i, j)
                if b > 0.0 and b > theta_2:
                    G.add_edge(j, i)
    return G

def genWDGraphFromPureDirectedGraph(PGD, return_list, threshold=-1):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for n, nbrs in PGD.adj.items():
        S1 = return_list[n]
        for nbr, eattr in nbrs.items():
            S2 = return_list[nbr]
            corr = S1.corr(S2)
            if corr >= threshold: G.add_edge(n, nbr, corr=corr)
    return G

# function for generating PCGD
# generate partial correlation directed graph
def genPartCorrGraph(UGD):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for n, nbrs in UGD.adj.items():
        for nbr, eattr in nbrs.items():
            G.add_edge(n, nbr, weight=eattr['corr'])
    return G

# function for generating ECGU
# generate entire correlation undirected graph
def genEntCorrGraph():
    G = nx.Graph()
    G.add_nodes_from(lst_tickers_stp)
    for i in range(LENTCKR):
        S1 = df_stock_abr[lst_tickers_stp[i]]
        for j in range(i+1, LENTCKR):
            S2 = df_stock_abr[lst_tickers_stp[j]]
            G.add_edge(n, nbr, weight=S1.corr(S2))
    return G

def rmvEdgeAttrOfGraph(WG):
    G = WG.copy()
    for n, nbrs in G.adj.items():
        for nbr, eattr in nbrs.items():
            if 'corr' in eattr: del eattr['corr']
    return G

def rmvIndepNodesFromGraph(WholeG):
    G = WholeG.copy()
    for n in WholeG.nodes():
        if WholeG.degree(n) == 0: G.remove_node(n)
    return G

def _distance_matrix(L):
    Dmax = L//2

    D = list(range(Dmax+1))
    D += D[-2+(L%2):0:-1]

    return circulant(D)/Dmax

def _pd(d, p0, beta):
    return beta*p0 + (d <= p0)*(1-beta)

def watts_strogatz(L, p0, beta, directed=False, rngseed=1):
    """
    Watts-Strogatz model of a small-world network

    This generates the full adjacency matrix, which is not a good way to store
    things if the network is sparse.

    Parameters
    ----------
    L        : int
               Number of nodes.
 
    p0       : float
               Edge density. If K is the average degree then p0 = K/(L-1).
               For directed networks "degree" means out- or in-degree.
 
    beta     : float
               "Rewiring probability."
 
    directed : bool
               Whether the network is directed or undirected.
 
    rngseed  : int
               Seed for the random number generator.
 
    Returns
    -------
    A        : (L, L) array
               Adjacency matrix of a WS (potentially) small-world network.
 
    """
    rng = np.random.RandomState(rngseed)

    d = _distance_matrix(L)
    p = _pd(d, p0, beta)

    if directed:
        A = 1*(rng.random_sample(p.shape) < p)
        np.fill_diagonal(A, 0)
    else:
        upper = np.triu_indices(L, 1)
 
        A          = np.zeros_like(p, dtype=int)
        A[upper]   = 1*(rng.rand(len(upper[0])) < p[upper])
        A.T[upper] = A[upper]

    return A

dct_title_amt = dict(df_codes_and_title['Title'].value_counts())
dct_BEA_amt = dict(df_codes_and_title['BEA'].value_counts())

def genEIODirectMatrix(directType, tradeoff = True, to_log = True):
    global dct_title_amt
    global EIO_industry_title_list
    industry_list = EIO_industry_title_list
    EIO_direct_matrix = np.matrix(
        np.zeros(
            (len(industry_list),
             len(industry_list)),
            dtype=np.float64))
    if directType is 'requirements':
        for i in range(len(industry_list)):
            for j in range(len(industry_list)):
                if EIO_matrix[i, j] > 0:
                    m = np.float64(EIO_matrix[i, j]) / EIO_matrix[-1, j]
                    if tradeoff and industry_list[i] in dct_title_amt:
                        m /= dct_title_amt[industry_list[i]]
                    EIO_direct_matrix[i, j] = logarise(m) if to_log else m
    elif directType is 'demands':
        for i in range(len(industry_list)):
            for j in range(len(industry_list)):
                if EIO_matrix[i, j] > 0:
                    m = np.float64(EIO_matrix[i, j]) / EIO_matrix[i, -1]
                    if tradeoff and industry_list[j] in dct_title_amt:
                        m /= dct_title_amt[industry_list[j]]
                    EIO_direct_matrix[i, j] = logarise(m) if to_log else m
    else: return None
    return EIO_direct_matrix

def getAllMatrixContent(mat):
    arr = []
    for m in mat:
        arr = np.append(arr, np.array(m)[0])
    return arr

def getNonzeroMatrixContent(mat):
    arr = []
    for m in mat:
        ar_m = np.array(m)[0]
        ar_m = ar_m[ar_m!=0]
        arr = np.append(arr, ar_m)
    return arr

def genEdgeDensity(lst, bins=100):
    theta_thresholds = np.linspace(np.floor(min(lst)*10.0)/10.0, np.ceil(max(lst)*10.0)/10.0, bins)
    edge_densities = []
    n = 0
    LENLST_FLOAT = np.float(len(lst))
    for theta in theta_thresholds:
        n += 1
        #print(n, end=' ')
        edge_densities.append(sum(corr >= theta for corr in lst)/LENLST_FLOAT)
    return theta_thresholds, edge_densities

def logarise(n): return 0.0 if n == 0 else -1.0/np.log10(n)

def combineThresholds(thresholds_1, thresholds_2, mat_1, mat_2):
    LENMAT = mat_1.shape[0]
    df = pd.DataFrame(
        index=range(len(thresholds_1)*len(thresholds_2)),
        columns=['theta_DR', 'theta_DD', 'no_directions'])
    idx = 0
    print(len(thresholds_1), end='')
    for t1 in thresholds_1:
        print('.', end='')
        exceeded = False
        for t2 in thresholds_2:
            cnt = 0
            if not exceeded:
                for i in range(LENMAT):
                    for j in range(LENMAT):
                        a = mat_1[i, j]
                        b = mat_2[i, j]
                        if (a > 0.0 and a > t1) or (b > 0.0 and b > t2):
                            cnt += 1
                if cnt == 0: exceeded = True
            df.iloc[idx,:] = [t1, t2, cnt]
            idx += 1
    return df

def combineThresholdsOfEIOAndCorrForAmtOfEdges(thresholds_eio, thresholds_corr, FG):
    global lst_tickers_stp
    df = pd.DataFrame(
        index=range(len(thresholds_eio)*len(thresholds_corr)),
        columns=['theta_EIO', 'theta_corr', 'no_edges'])
    idx = 0
    numrow = 0
    for t1 in thresholds_eio:
        print('[%s]' % numrow)
        exceeded = False
        for t2 in thresholds_corr:
            cnt = 0
            if not exceeded:
                for n, nbrs in FG.adj.items():
                    for nbr, eattr in nbrs.items():
                        if ('direct_requirement' in eattr and eattr['direct_requirement'] > t1) or ('direct_demand' in eattr and (eattr['direct_demand'] > t1)):
                            if eattr['corr'] > t2: cnt += 1
                if cnt == 0: exceeded = True
            print(cnt, end=' ')
            df.iloc[idx,:] = [t1, t2, cnt]
            idx += 1
        numrow += 1
    return df

def combineThresholdsOfEIOAndCorrForIsWeaklyConnected(thresholds_eio, thresholds_corr, FG):
    global lst_tickers_stp
    df = pd.DataFrame(
        index=range(len(thresholds_eio)*len(thresholds_corr)),
        columns=['EIO', 'corr', 'is_weakly_connected'])
    idx = 0
    for t1 in thresholds_eio:
        for t2 in thresholds_corr:
            G = nx.DiGraph()
            G.add_nodes_from(lst_tickers_stp)
            for n, nbrs in FG.adj.items():
                for nbr, eattr in nbrs.items():
                    if ('direct_requirement' in eattr and eattr['direct_requirement'] > t1) or ('direct_demand' in eattr and (eattr['direct_demand'] > t1)):
                        if eattr['corr'] > t2: G.add_edge(n, nbr)
            G = rmvIndepNodesFromGraph(G)
            print(G.number_of_nodes(), end=' ')
            if G.number_of_nodes() > 0:
                is_weakly_c = nx.is_weakly_connected(G)  
                print(is_weakly_c, end=' ')
                df.iloc[idx,:] = [t1, t2, is_weakly_c]
            else:
                print('Nill', end=' ')
                df.iloc[idx,:] = [t1, t2, False]
            idx += 1
    return df

def continueCombineThresholdsOfEIOAndCorrForIsWeaklyConnected(thresholds_eio, thresholds_corr, FG, start_point):
    global df
    i = 0
    idx = start_point
    cnt = 0
    for t1 in thresholds_eio:
        print('[%s]' % cnt)
        for t2 in thresholds_corr:
            if i < start_point:
                i += 1
                continue
            G = nx.DiGraph()
            G.add_nodes_from(lst_tickers_stp)
            for n, nbrs in FG.adj.items():
                for nbr, eattr in nbrs.items():
                    if ('direct_requirement' in eattr and eattr['direct_requirement'] > t1) or ('direct_demand' in eattr and (eattr['direct_demand'] > t1)):
                        if eattr['corr'] > t2: G.add_edge(n, nbr)
            G = rmvIndepNodesFromGraph(G)
            print(G.number_of_nodes(), end=' ')
            if G.number_of_nodes() > 0:
                is_weakly_c = nx.is_weakly_connected(G)  
                print(is_weakly_c, end=' ')
                df.iloc[idx,:] = [t1, t2, is_weakly_c]
            else:
                print('Nill', end=' ')
                df.iloc[idx,:] = [t1, t2, False]
            idx += 1
        cnt += 1
        
FILE_EIO_2016 = ROOTPATH + '/Source/lxl/EIO_2016.csv'
EIO_matrix = np.matrix(np.genfromtxt(open(FILE_EIO_2016, 'rb'), delimiter=',', skip_header=2))
EIO_industry_BEA_code_list = list(pd.read_csv(FILE_EIO_2016, nrows=0).columns)[:-2]
EIO_industry_title_list = list(pd.read_csv(FILE_EIO_2016, skiprows=1).columns)[:-2]
FILE_STOCK_ABR = ROOTPATH + r'/Source/DF_STOCK_ABR.csv'
df_stock_abr = pd.read_csv(FILE_STOCK_ABR).set_index('Date')
df_stock_normal_return = pd.DataFrame(index=df_stock_abr.index, columns=df_stock_abr.columns)
for i in DICT_STP: df_stock_normal_return[i] = DICT_STP[i]['log_return']
    
EIO_direct_requirements_matrix = genEIODirectMatrix('requirements', tradeoff=True, to_log=True)
EIO_direct_demands_matrix = genEIODirectMatrix('demands', tradeoff=True, to_log=True)

ar_all_DR_Mat = getAllMatrixContent(EIO_direct_requirements_matrix)
ar_all_DD_Mat = getAllMatrixContent(EIO_direct_demands_matrix)

ar_all_DR_trans = [i for i in ar_all_DR_Mat]
ar_all_DD_trans = [i for i in ar_all_DD_Mat]

theta_thresholds_DR_all, edge_densities_DR_all = genEdgeDensity(ar_all_DR_trans, 100)
theta_thresholds_DD_all, edge_densities_DD_all = genEdgeDensity(ar_all_DD_trans, 100)

ar_nonzero_DR_Mat = getNonzeroMatrixContent(EIO_direct_requirements_matrix)
ar_nonzero_DD_Mat = getNonzeroMatrixContent(EIO_direct_demands_matrix)

ar_nonzero_DR_trans = [i for i in ar_nonzero_DR_Mat]
ar_nonzero_DD_trans = [i for i in ar_nonzero_DD_Mat]

theta_thresholds_DR, edge_densities_DR = genEdgeDensity(ar_nonzero_DR_trans, 100)
theta_thresholds_DD, edge_densities_DD = genEdgeDensity(ar_nonzero_DD_trans, 100)

b1 = np.append(1.0, edge_densities_DR_all)
b1[1] = b1[2]
b2 = np.append(0, theta_thresholds_DR_all)

x_dashline = 0.136
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.set_xlabel('threshold')
ax.set_ylabel('Transaction density')
ax.set_xlim(left=-0.05, right=1.2)
ax.set_title('Normalised Direct Requirement', fontsize='large')
dashed_line = Line2D([x_dashline, x_dashline], [-1.05, 1.05], linestyle = '--', linewidth = 1, color = [0.3,0.3,0.3], zorder = 1, transform = ax.transData)
ax.lines.append(dashed_line)
ax.plot(b2, b1, color='blue', lw=2)
ax = fig.add_subplot(2,1,2)
ax.set_xlabel('threshold')
ax.set_ylabel('Transaction density')
ax.set_title('Normalised Direct Demand', fontsize='large')
dashed_line = Line2D([x_dashline, x_dashline], [-0.05, 1.05], linestyle = '--', linewidth = 1, color = [0.3,0.3,0.3], zorder = 1, transform = ax.transData)
ax.lines.append(dashed_line)
ax.set_xlim(left=-0.05, right=1.2)
ax.plot(b2, b1, color='blue', lw=2)
fig.tight_layout()

df_combined_thresholds = combineThresholds(
    theta_thresholds_DR,
    theta_thresholds_DD,
    EIO_direct_requirements_matrix,
    EIO_direct_demands_matrix)

pt = df_combined_thresholds.pivot_table(index='theta_DR', columns='theta_DD', values='no_directions', aggfunc=np.sum)
f, ax = plt.subplots(figsize = (10, 4))
sns.heatmap(pt.iloc[:30,:20], cmap='rainbow', linewidths = 0.05, ax = ax)
ax.set_title('Amounts of directions per DR-threshold and DD-threshold')
ax.set_xlabel('theta_DD')
ax.set_ylabel('theta_DR')

def genFullGraph(stock_return_df):
    global lst_tickers_stp
    global EIO_industry_BEA_code_list
    global EIO_direct_requirements_matrix
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for i in lst_tickers_stp:
        #print(i, end=' ')
        matidx_i = EIO_industry_BEA_code_list.index(df_codes_and_title.loc[i, 'BEA'])
        S1 = stock_return_df[i]
        for j in lst_tickers_stp:
            if i != j:
                matidx_j = EIO_industry_BEA_code_list.index(df_codes_and_title.loc[j, 'BEA'])
                if EIO_direct_requirements_matrix[matidx_i, matidx_j] > 0:
                    G.add_edge(i, j, direct_requirement = EIO_direct_requirements_matrix[matidx_i, matidx_j])
                if EIO_direct_demands_matrix[matidx_j, matidx_i] > 0:
                    G.add_edge(i, j, direct_demand = EIO_direct_demands_matrix[matidx_j, matidx_i])
                if G.has_edge(i, j): G.add_edge(i, j, corr=S1.corr(stock_return_df[j]))
    return G

FullG = genFullGraph(df_stock_normal_return)
direct_requirements_CN = []
direct_demands_CN = []
for n, nbrs in FullG.adj.items():
    for nbr, eattr in nbrs.items():
        if 'direct_requirement' in eattr.keys():
            direct_requirements_CN.append(eattr['direct_requirement'])
        if 'direct_demand' in eattr.keys():
            direct_demands_CN.append(eattr['direct_demand'])
            
corr_coef_CN = []
for n, nbrs in FullG.adj.items():
    for nbr, eattr in nbrs.items():
        corr_coef_CN.append(eattr['corr'])

plt.hist(corr_coef_CN, density=1, bins=260, histtype='bar')
plt.axis([-0.5, 1, 0, 3.2])
plt.xlabel('correlation')
plt.ylabel('p(correlation)')

describe(corr_coef_CN)

corr_mean = np.mean(corr_coef_CN)
corr_std = np.std(corr_coef_CN)
corr_coef_CN_00 = [(i - corr_mean)/corr_std for i in corr_coef_CN]

stats.probplot(corr_coef_CN, dist='norm', plot=pylab)
pylab.show()

sm.qqplot(np.array(corr_coef_CN_00), line='45',)
pylab.show()

ss.kstest(corr_coef_CN_00, 'norm')

theta_thresholds_corr, edge_densities_corr = genEdgeDensity(corr_coef_CN, 100)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('threshold')
ax.set_ylabel('Edge density')
ax.set_title('Correlation Coefficient', fontsize='large')
ax.plot(theta_thresholds_corr, edge_densities_corr, color='blue', lw=2)
fig.tight_layout()

recalc = False
npzfile_name = data_dir + '/pt_cteac_0719.npz'
pt_cteac = None
if recalc == True:
    df = pd.DataFrame(
        index=range(len(theta_thresholds_DR)*len(theta_thresholds_corr)),
        columns=['EIO', 'corr', 'is_weakly_connected'])
    continueCombineThresholdsOfEIOAndCorrForIsWeaklyConnected(
        theta_thresholds_DR, theta_thresholds_corr, FullG, 0)
    df_cteac = df.copy()
    pt_cteac = df_cteac.pivot_table(
        index = 'EIO', columns='corr',
        values = 'is_weakly_connected', aggfunc=np.sum)
    outfile = open(npzfile_name, 'wb')
    np.savez(outfile, ar_cteac=pt_cteac, col=pt_cteac.columns, ind=pt_cteac.index)
    outfile.close()
else:
    infile = open(npzfile_name, 'rb')
    npzfile = np.load(infile)
    infile.close()
    ar_cteac = npzfile['ar_cteac']
    pt_cteac = pd.DataFrame(ar_cteac)
    pt_cteac.columns = npzfile['col']
    pt_cteac.index = npzfile['ind']
    
f, ax = plt.subplots(figsize = (10, 4))
sns.heatmap(pt_cteac.iloc[5:50,20:90], cmap='rainbow', linewidths = 0.05, ax = ax)
ax.set_title('Amounts of directions per DR-threshold and DD-threshold')
ax.set_xlabel('corr')
ax.set_ylabel('EIO');

recalc = True
npzfile_name = data_dir + '/pt_toeacfaoe_0719.npz'
pt_toeacfaoe = None
if recalc == True:
    df_toeacfaoe = combineThresholdsOfEIOAndCorrForAmtOfEdges(
        theta_thresholds_DR, theta_thresholds_corr, FullG)
    pt_toeacfaoe = df_toeacfaoe.pivot_table(
        index = 'theta_EIO', columns='theta_corr',
        values = 'no_edges', aggfunc=np.sum)
    outfile = open(npzfile_name, 'wb')
    np.savez(outfile, ar_toeacfaoe=pt_toeacfaoe,
             col=pt_toeacfaoe.columns, ind=pt_toeacfaoe.index)
    outfile.close()
else:
    infile = open(npzfile_name, 'rb')
    npzfile = np.load(infile)
    infile.close()
    ar_toeacfaoe = npzfile['ar_toeacfaoe']
    pt_toeacfaoe = pd.DataFrame(ar_cteac)
    pt_toeacfaoe.columns = npzfile['col']
    pt_toeacfaoe.index = npzfile['ind']
    
pt_toeacfaoe.index = [round(i, 4) for i in pt_toeacfaoe.index]
pt_toeacfaoe.columns = [round(i, 4) for i in pt_toeacfaoe.columns]

f, ax = plt.subplots(figsize = (10, 4))
sns.heatmap(pt_toeacfaoe.iloc[:24,:81], cmap='gist_ncar', linewidths = 0.05, ax = ax)
ax.set_xlabel(r'$\theta_{corr}$')
ax.set_ylabel(r'$\theta_{EIO}$');

threshold_eio = 0.29225
threshold_corr = 0.378705
DiUnwtG = genPureDedirectedGraph(
    threshold_eio, #0.35
    threshold_eio, #0.35
    EIO_direct_requirements_matrix,
    EIO_direct_demands_matrix)

G = genWDGraphFromPureDirectedGraph(DiUnwtG, df_stock_normal_return, threshold_corr)
nonodes = G.number_of_nodes()
noedges = G.number_of_edges()
DiG_pureedge = rmvEdgeAttrOfGraph(G)
DiG_connected = rmvIndepNodesFromGraph(DiG_pureedge)

def genConventionalGraph(theta, return_list):
    global LENTCKR
    global lst_tickers_stp
    G = nx.Graph()
    G.add_nodes_from(lst_tickers_stp)
    for i in range(LENTCKR):
        T1 = lst_tickers_stp[i]
        S1 = return_list[T1]
        for j in range(i+1, LENTCKR):
            T2 = lst_tickers_stp[j]
            S2 = return_list[T2]
            corr = S1.corr(S2)
            if corr > theta: G.add_edge(T1, T2, corr=corr)
    return G

G_conv = genConventionalGraph(0.4983, df_stock_normal_return)
G_conv.number_of_edges()

data = [d for n, d in G.out_degree()]
plt.hist(data, density=1, bins=50, histtype='bar');
fit = powerlaw.Fit(data)
fit.distribution_compare('power_law', 'lognormal')

fig4 = fit.plot_ccdf(linewidth = 2)
fit.power_law.plot_ccdf(ax = fig4, color = 'r', linestyle = '--');
fig4.set_xlabel('centrality')
fig4.set_ylabel('$p(X\geq x)$')
fig4.legend(('CCDF','Power-law fit'))
set_size(4,4,fig4)

data = [d for n, d in G_rd.out_degree()]
plt.hist(data, density=1, bins=50, histtype='bar');

stats.probplot([d for n, d in G_rd.out_degree()], dist='norm', plot=pylab)

fig = sns.distplot([d for n, d in G_rd.out_degree()])
fig.set_xlabel('out-degree')
fig.set_ylabel('p(out-degree)')

data = [d for n, d in G_ws_mat.out_degree()]
plt.hist(data, density=1, bins=50, histtype='barstacked');

data = [d for n, d in G_ws_mat.out_degree() if d > 0]
powerlaw.plot_pdf(data, linear_bins = False, color = 'b');

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
data = [d for n, d in G.out_degree() if d > 0]
powerlaw.plot_pdf(data, linear_bins = False, color = 'b')
ax.set_xlabel('out-degree')
ax.set_ylabel('p(out-degree)')
set_size(4,4,ax)

data = [d for n, d in G.in_degree() if d > 0]
powerlaw.plot_pdf(data, linear_bins = False, color = 'b');

data = [d for n, d in G.degree() if d > 0]
powerlaw.plot_pdf(data, linear_bins = False, color = 'b');

p0 = np.average([i[1] for i in G.out_degree()])/(nonodes-1)
ws_mat = watts_strogatz(L=nonodes, p0=p0, beta=0.5, directed=True)
G_ws_mat=nx.from_numpy_matrix(ws_mat, create_using=nx.DiGraph())
G_ws_mat.number_of_edges()

p = noedges / nonodes / (nonodes-1)
G_rd = nx.generators.gnp_random_graph(nonodes, p=p, directed=True)
G_rd.number_of_edges()

def calGlobalEfficiency(G, lst_nodes, N): #N = 0, 
    #if N == 0: N = G.number_of_nodes()
    #if lst_nodes == None: lst_nodes = G.nodes()
    shortest_path = nx.shortest_path(G)
    acc = 0.0
    for i in lst_nodes:
        for j in lst_nodes:
            if i != j and (j in shortest_path[i]):
                acc += 1.0/(len(shortest_path[i][j])-1)
    return acc/N/(N-1)

def calLocalEfficiency(G):
    UndiG = G.to_undirected()
    lst_nodes = G.nodes()
    acc = 0.0
    for i in lst_nodes:
        print(i, end='')
        nodes_g = list(UndiG[i])
        n = len(nodes_g)
        if n > 0:
            #nodes_g.append(i)
            print('(%s)'%n, end=' ')
            acc += calGlobalEfficiency(G.subgraph(nodes_g), nodes_g, n)
    return acc/G.number_of_nodes()

calGlobalEfficiency(DiG_connected, G.number_of_nodes())
calGlobalEfficiency(G_ws_mat)
calLocalEfficiency(G)

def detectCommunityForDirectedGraph(G):
    lst_node = list(G.nodes)
    NONODES = G.number_of_nodes()
    NOEDGES = G.number_of_edges()
    NOINDEGREES = G.in_degree()
    NOOUTDEGREES = G.out_degree()
    overall_asgn = [(0, 0)] * NONODES
    
    def getNodeSpace(node_space, upd_asgn, val):
        return [node_space[i] for i in range(len(upd_asgn)) if upd_asgn[i] == val]
    '''
    def fineTune(node_space, upd_asgn):
        nonlocal G
        nonlocal overall_asgn
        nonlocal NONODES
        for i in upd_asgn:
            for j in node_space:
'''
    def interateBisection(mod_mat, node_space, generation_mark):
        nonlocal G
        nonlocal overall_asgn
        upd_asgn = subdivideCommunities(mod_mat)
        # todo: fine-tune
        
        if len(np.unique(upd_asgn)) == 1: return
        delta_Q = calDeltaQ(upd_asgn, mod_mat)
        print('calDeltaQ: %s' % delta_Q)
        if delta_Q < 0: return
        node_space_1 = getNodeSpace(node_space, upd_asgn, -1)
        if len(node_space_1) == 0: return
        updCommunityAssignment(node_space_1, upd_asgn, generation_mark)
        print('genGeneralisedModularityMatrix_1: %s:%s:%s' % (time.localtime()[3],time.localtime()[4],time.localtime()[5]))
        mod_mat_1 = genGeneralisedModularityMatrix(node_space_1)
        interateBisection(mod_mat_1, node_space_1, generation_mark+1)
        node_space_2 = getNodeSpace(node_space, upd_asgn, 1)
        if len(node_space_2) == 0: return
        updCommunityAssignment(node_space_2, upd_asgn, generation_mark)
        print('genGeneralisedModularityMatrix_2: %s:%s:%s' % (time.localtime()[3],time.localtime()[4],time.localtime()[5]))
        mod_mat_2 = genGeneralisedModularityMatrix(node_space_2)
        interateBisection(mod_mat_2, node_space_2, generation_mark+1)
        return
    
    def genGeneralisedModularityMatrix(node_space):# Subgraph
        nonlocal G
        nonlocal lst_node
        nonlocal NOEDGES
        nonlocal NOINDEGREES
        nonlocal NOOUTDEGREES
        nonlocal overall_asgn
        LENNODESPECE = len(node_space)
        mod_mat = np.matrix(np.zeros((LENNODESPECE, LENNODESPECE), dtype=np.float64))
        for i in range(LENNODESPECE):
            print(i, end=' ')
            tckr_i = lst_node[node_space[i]]
            for j in range(LENNODESPECE):
                tckr_j = lst_node[node_space[j]]
                Bij = G.has_edge(tckr_j, tckr_i) - NOINDEGREES[tckr_i] * NOOUTDEGREES[tckr_j] / NOEDGES
                if overall_asgn[node_space[i]] == overall_asgn[node_space[j]]:
                    Ck = 0.0
                    for k in node_space:
                        tckr_k = lst_node[k]
                        Ck += G.has_edge(tckr_k, tckr_i) + G.has_edge(tckr_i, tckr_k) - (NOINDEGREES[tckr_i] * NOOUTDEGREES[tckr_k] + NOINDEGREES[tckr_k] * NOOUTDEGREES[tckr_i]) / NOEDGES
                    mod_mat[i, j] = Bij - Ck / 2.0
                else: mod_mat[i, j] = Bij
        print('/')
        return mod_mat
    
    def updCommunityAssignment(node_space, upd_asgn, generation_mark):
        nonlocal overall_asgn
        global asgn_history
        inreval_1 = 0
        inreval_2 = 0
        lst_gener_asgn = []
        for asgn in overall_asgn:
            if asgn[0] == generation_mark:
                lst_gener_asgn.append(asgn[1])
        for i in range(len(node_space)):
            if i not in lst_gener_asgn:
                inreval_1 = i
                break
        if upd_asgn.count(1) == 0: inreval_2 = inreval_1
        else:
            for i in range(len(node_space)):
                if i not in lst_gener_asgn:
                    inreval_2 = i
                    break
        for i in range(len(node_space)):
            if upd_asgn[i] == 1: overall_asgn[node_space[i]] = (generation_mark, inreval_1)
            if upd_asgn[i] == -1: overall_asgn[node_space[i]] = (generation_mark, inreval_2)
        #print(overall_asgn)
        asgn_history.append(overall_asgn.copy())
    
    def subdivideCommunities(mod_mat):
        sym_mat = mod_mat + mod_mat.T
        w, v = np.linalg.eigh(sym_mat)
        eigv = v[:, len(w)-1]
        return [np.sign(v.tolist()[0][0]) for v in eigv]
        
    def calDirectedGraphModularity(assignment):
        nonlocal G
        nonlocal lst_node
        nonlocal NONODES
        nonlocal NOEDGES
        nonlocal NOINDEGREES
        nonlocal NOOUTDEGREES
        Q = 0.0
        for i in range(NONODES):
            for j in range(NONODES):
                if assignment[i] == assignment[j]:
                    Q += G.has_edge(lst_node[j], lst_node[i]) - NOINDEGREES[lst_node[i]] * NOOUTDEGREES[lst_node[j]] / NOEDGES
        return Q / NOEDGES
    
    def calDeltaQ(upd_asgn, Bg):
        nonlocal NOEDGES
        sg = np.matrix(upd_asgn)
        return 0.25/NOEDGES*np.dot(np.dot(sg, (Bg+Bg.T)), sg.T)[0,0]
    
    MODMAT = np.matrix(np.zeros((NONODES, NONODES), dtype=np.float64))
    for i in range(NONODES):
        for j in range(NONODES):
            MODMAT[i, j] = G.has_edge(lst_node[j], lst_node[i]) - NOINDEGREES[lst_node[i]] * NOOUTDEGREES[lst_node[j]] / NOEDGES
    interateBisection(MODMAT, list(np.arange(NONODES)), 1)
    return overall_asgn, calDirectedGraphModularity(overall_asgn)

def calModularity(assignment):
    global G
    global lst_node
    global NONODES
    global NOEDGES
    global NOINDEGREES
    global NOOUTDEGREES
    Q = 0.0
    for i in range(NONODES):
        for j in range(NONODES):
            if assignment[i] == assignment[j]:
                Q += G.has_edge(lst_node[j], lst_node[i]) - NOINDEGREES[lst_node[i]] * NOOUTDEGREES[lst_node[j]] / NOEDGES
    return Q / NOEDGES

lst_node = list(G.nodes())
NONODES = G.number_of_nodes()
NOEDGES = G.number_of_edges()
NOINDEGREES = G.in_degree()
NOOUTDEGREES = G.out_degree()

over_best_asgn = [0] * len(lst_tickers_stp)
for i in range(len(lst_tickers_stp)):
    over_best_asgn[i] = int(G.node[lst_tickers_stp[i]]['community'])
    
best_asgn = over_best_asgn.copy()
origin_mod = calModularity(best_asgn)
for i in range(len(best_asgn)):
    asgn = best_asgn[i]
    for uniq in uniq_asgn:
        if asgn != uniq:
            best_asgn[i] = uniq
            new_mod = calModularity(best_asgn)
            if new_mod > origin_mod:
                origin_mod = new_mod
                asgn = uniq
            else: best_asgn[i] = asgn

lst_tckr_nonzero = [i[0] for i in G.degree if i[1]>0]
rdmchosen_tickers = np.random.choice(lst_tckr_nonzero, 1, replace=False)
nonzerodeg_subG = G.subgraph(lst_tckr_nonzero).copy()

asgn_history = []
overall_asgn, modularity = detectCommunityForDirectedGraph(nonzerodeg_subG)

stp_group_tckr_index = sri_overall_asgn[sri_overall_asgn == v_c.index[0]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_1 = pd.Series(stp_group_indcode).value_counts()
stp_group_tckr_index = sri_overall_asgn[sri_overall_asgn == v_c.index[1]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_2 = pd.Series(stp_group_indcode).value_counts()
stp_group_tckr_index = sri_overall_asgn[sri_overall_asgn == v_c.index[2]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_3 = pd.Series(stp_group_indcode).value_counts()
stp_group_tckr_index = sri_overall_asgn[sri_overall_asgn == v_c.index[3]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_4 = pd.Series(stp_group_indcode).value_counts()
stp_group_tckr_index = sri_overall_asgn[sri_overall_asgn == v_c.index[4]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_5 = pd.Series(stp_group_indcode).value_counts()

stp_group_tckr_index = []
for i in range(5,11): stp_group_tckr_index += sri_overall_asgn[sri_overall_asgn == v_c.index[i]].index.tolist()
stp_group_tckr = [lst_tickers_stp[i] for i in stp_group_tckr_index]
stp_group_indcode = getIndustryCodeByStockCode(stp_group_tckr, code_type='Title')
sri_community_6 = pd.Series(stp_group_indcode).value_counts()

uniq_ind = np.unique(df_codes_and_title.Title)

lst_community_1 = []
lst_community_2 = []
lst_community_3 = []
lst_community_4 = []
lst_community_5 = []
lst_community_6 = []
for ind in uniq_ind:
    if ind in sri_community_1.index: lst_community_1.append(sri_community_1[ind]) #/ sum(sri_overall_asgn == v_c.index[0]))
    else: lst_community_1.append(0)
    if ind in sri_community_2.index: lst_community_2.append(sri_community_2[ind]) #/ sum(sri_overall_asgn == v_c.index[1]))
    else: lst_community_2.append(0)
    if ind in sri_community_3.index: lst_community_3.append(sri_community_3[ind]) #/ sum(sri_overall_asgn == v_c.index[2]))
    else: lst_community_3.append(0)
    if ind in sri_community_4.index: lst_community_4.append(sri_community_4[ind]) #/ sum(sri_overall_asgn == v_c.index[3]))
    else: lst_community_4.append(0)
    if ind in sri_community_5.index: lst_community_5.append(sri_community_5[ind]) #/ sum(sri_overall_asgn == v_c.index[4]))
    else: lst_community_5.append(0)
    if ind in sri_community_6.index: lst_community_6.append(sri_community_6[ind]) #/ sum(sri_overall_asgn == v_c.index[5]))
    else: lst_community_6.append(0)
        
lst2_sector = [None] * len(uniq_ind)
for i in range(len(uniq_ind)):
    lst2_sector[i] = []
    if uniq_ind[i] in sri_community_1.index: lst2_sector[i].append(sri_community_1[uniq_ind[i]])
    else: lst2_sector[i].append(0)
    if uniq_ind[i] in sri_community_2.index: lst2_sector[i].append(sri_community_2[uniq_ind[i]])
    else: lst2_sector[i].append(0)
    if uniq_ind[i] in sri_community_3.index: lst2_sector[i].append(sri_community_3[uniq_ind[i]])
    else: lst2_sector[i].append(0)
    if uniq_ind[i] in sri_community_4.index: lst2_sector[i].append(sri_community_4[uniq_ind[i]])
    else: lst2_sector[i].append(0)
    if uniq_ind[i] in sri_community_5.index: lst2_sector[i].append(sri_community_5[uniq_ind[i]])
    else: lst2_sector[i].append(0)
    if uniq_ind[i] in sri_community_6.index: lst2_sector[i].append(sri_community_6[uniq_ind[i]])
    else: lst2_sector[i].append(0)
        
idx = np.arange(6)
plt.figure(figsize=(15,15))

colors = list(dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS).values())
idx = np.arange(6)
plt.figure(figsize=(10,10))
p = []
for i in range(len(uniq_ind)):
    bottom = [0] * 6
    for j in range(i): bottom = [a+b for a, b in zip(bottom, np.array(lst2_sector[j]))]
    p.append(plt.bar(idx, lst2_sector[i], bottom=bottom, color=colors[np.random.randint(len(colors))]))
plt.xticks(idx, ['C1','C2','C3','C4','C5','Others'])

idx = np.arange(len(uniq_ind))
plt.figure(figsize=(15,15))
p1 = plt.bar(idx, lst_community_1, color=(0.85, 0.5176, 1))
p2 = plt.bar(idx, lst_community_2, bottom=lst_community_1, color=(0.553, 0.753, 0.0863))
p3 = plt.bar(idx, lst_community_3,
             bottom=np.array(lst_community_1)+np.array(lst_community_2), color=(0.4157, 0.7608, 1))
p4 = plt.bar(idx, lst_community_4,
             bottom=np.array(lst_community_1)+np.array(lst_community_2)+np.array(lst_community_3), color=[0.937255,0.851,0.25333])
p5 = plt.bar(idx, lst_community_5,
             bottom=np.array(lst_community_1)+np.array(lst_community_2)+np.array(lst_community_3)+np.array(lst_community_4), color=[0.898,0.5294,0.1294])
p6 = plt.bar(idx, lst_community_6,
             bottom=np.array(lst_community_1)+np.array(lst_community_2)+np.array(lst_community_3)+np.array(lst_community_4)+np.array(lst_community_5), color=[0.283,0.2823,0.282])
plt.xticks(idx, uniq_ind, rotation=90)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('Community 1', 'Community 2', 'Community 3', 'Community 4', 'Community 5', 'Unclustered'));
