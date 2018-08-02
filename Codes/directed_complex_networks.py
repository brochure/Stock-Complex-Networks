from __future__ import division
import sys, csv, time, requests, statsmodels, math
from statsmodels.tsa.stattools import coint, adfuller, grangercausalitytests
import statsmodels.api as sm
from scipy import stats
import scipy.special
from scipy.stats import describe
from contextlib import contextmanager
from datetime import datetime, timedelta
from dateutil.parser import parse
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
import numpy as np
from core import SendEmail, ROOTPATH, sri_SP500_log_return
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL, df_codes_and_title, DICT_STP, getIndustryCodeByStockCode

def genEIODirectMatrix(directType):
    EIO_direct_matrix = np.matrix(
        np.zeros(
            (len(EIO_industry_BEA_code_list),
             len(EIO_industry_BEA_code_list)),
            dtype=np.float64
        ))
    if directType is 'requirements':
        for i in range(len(EIO_industry_BEA_code_list)):
            for j in range(len(EIO_industry_BEA_code_list)):
                if EIO_matrix[i, j] > 0:
                    EIO_direct_matrix[i, j] = np.float64(EIO_matrix[i, j]) / EIO_matrix[-1, j]
    elif directType is 'demands':
        for i in range(len(EIO_industry_BEA_code_list)):
            for j in range(len(EIO_industry_BEA_code_list)):
                if EIO_matrix[i, j] > 0:
                    EIO_direct_matrix[i, j] = np.float64(EIO_matrix[i, j]) / EIO_matrix[i, -1]
    else: return None
    return EIO_direct_matrix

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
    for theta in theta_thresholds:
        n += 1
        edge_densities.append(sum(corr >= theta for corr in lst)/np.float(len(lst)))
    return theta_thresholds, edge_densities

def logarise(n): return -1.0/np.log10(n)

def combineThresholds(thresholds_1, thresholds_2, mat_1, mat_2):
    LENMAT = mat_1.shape[0]
    df = pd.DataFrame(
        index=range(len(thresholds_1)*len(thresholds_2)),
        columns=['theta_DR', 'theta_DD', 'no_directions'])
    idx = 0
    for t1 in thresholds_1:
        print(t1, end=', ')
        exceeded = False
        for t2 in thresholds_2:
            cnt = 0
            if not exceeded:
                for i in range(LENMAT):
                    for j in range(LENMAT):
                        a = mat_1[i, j]
                        b = mat_2[i, j]
                        if (a > 0.0 and logarise(a) > t1) or (b > 0.0 and logarise(b) > t2):
                            cnt += 1
                if cnt == 0: exceeded = True
            df.iloc[idx,:] = [t1, t2, cnt]
            idx += 1
    return df

'''
X = sm.add_constant(df_combined_thresholds.iloc[:,0:2])
y = df_combined_thresholds.no_directions
model = LinearRegression()
model.fit(X=X, y=y)
model.coef_
model.intercept_

# abnormal return
df_stock_abr_coef = pd.DataFrame(index=lst_tickers_stp, columns=['beta_0','beta_1'])
X = sm.add_constant(sri_SP500_log_return)
for tckr in lst_tickers_stp:
    y = DICT_STP[tckr]['log_return']
    model = sm.OLS(y,X)
    results = model.fit()
    df_stock_abr_coef.loc[tckr, 'beta_0'] = results.params['const']
    df_stock_abr_coef.loc[tckr, 'beta_1'] = results.params['Close']

df_stock_abr = pd.DataFrame(index=sri_SP500_log_return.index, columns=lst_tickers_stp)
for tckr in lst_tickers_stp:
     df_stock_abr[tckr] = DICT_STP[tckr]['log_return'] - df_stock_abr_coef.loc[tckr, 'beta_0'] - df_stock_abr_coef.loc[tckr, 'beta_1'] * sri_SP500_log_return

df_stock_abr.to_csv(ROOTPATH + r'/Stock_DWCN/Source/DF_STOCK_ABR.csv')
'''
###------------ Read Files ------------###
FILE_EIO_2016 = ROOTPATH + '/Stock_DWCN/Source/lxl/EIO_2016.csv'
EIO_matrix = np.matrix(np.genfromtxt(open(FILE_EIO_2016, 'rb'), delimiter=',', skip_header=2))
EIO_industry_BEA_code_list = list(pd.read_csv(FILE_EIO_2016, nrows=0).columns)[:-2]
FILE_STOCK_ABR = ROOTPATH + r'/Stock_DWCN/Source/DF_STOCK_ABR.csv'
df_stock_abr = pd.read_csv(FILE_STOCK_ABR).set_index('Date')
df_stock_normal_return = pd.DataFrame(index=df_stock_abr.index, columns=df_stock_abr.columns)
for i in DICT_STP: df_stock_normal_return[i] = DICT_STP[i]['log_return']
###------------ ---------- ------------###

complete_corr_coef_CN = [None] * LENTRGL
cnt = 0
for i in range(LENTCKR):
    S1 = df_stock_normal_return[lst_tickers_stp[i]]
    for j in range(i+1, LENTCKR):
        S2 = df_stock_normal_return[lst_tickers_stp[j]]
        complete_corr_coef_CN[cnt] = S1.corr(S2)
        cnt += 1

'''
n_ceiling = math.ceil(max([x for x in lst_trgl if x != float('inf')])) + 1
n_floor = math.floor(min([x for x in lst_trgl if x != float('-inf')])) - 1
lst_trgl = [n_ceiling if x == float('inf') else x for x in lst_trgl]
lst_trgl = [n_floor if x == float('-inf') else x for x in lst_trgl]
'''
'''
#G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
pos = nx.layout.shell_layout(TG)

node_sizes = [3 + 10 * i for i in range(len(TG))]
M = TG.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

nodes = nx.draw_networkx_nodes(TG, pos, node_size=node_sizes, node_color='blue')
edges = nx.draw_networkx_edges(TG, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_colors,
                               edge_cmap=plt.cm.Blues, width=2)
# set alpha value for each edge
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
pc.set_array(edge_colors)
plt.colorbar(pc)

ax = plt.gca()
ax.set_axis_off()
plt.show()
'''
###------------ Generate Direct Requirements Matrix ------------###
EIO_direct_requirements_matrix = genEIODirectMatrix('requirements')
EIO_direct_demands_matrix = genEIODirectMatrix('demands')

mat_max = 0
for m in EIO_direct_demands_matrix:
    max_t = max(np.array(m)[0])
    if max_t > mat_max and max_t < 1:
        mat_max = max_t
        print('mat_max: %s' % mat_max)
for i in range(EIO_direct_demands_matrix.shape[0]):
    ar_m = np.array(EIO_direct_demands_matrix[i])[0]
    for j in range(len(ar_m)):
        if ar_m[j] >= 1:
            EIO_direct_demands_matrix[i, j] = np.ceil(mat_max*10.0)/10.0
            print('EIO_direct_demands_matrix[%d, %d]: %s' % (i, j, EIO_direct_demands_matrix[i, j]))

print('Max in EIO_direct_requirements_matrix: %s' % max([max(np.array(m)[0]) for m in EIO_direct_requirements_matrix]))
print('Max in EIO_direct_demands_matrix: %s' % max([max(np.array(m)[0]) for m in EIO_direct_demands_matrix]))
## Edge density of the stock corr network for different values of the corr threshold


ar_nonzero_DR_Mat = getNonzeroMatrixContent(EIO_direct_requirements_matrix)
ar_nonzero_DD_Mat = getNonzeroMatrixContent(EIO_direct_demands_matrix)

ar_nonzero_DR_trans = [logarise(i) for i in ar_nonzero_DR_Mat]
ar_nonzero_DD_trans = [logarise(i) for i in ar_nonzero_DD_Mat]

theta_thresholds_DR, edge_densities_DR = genEdgeDensity(ar_nonzero_DR_trans, 100)
theta_thresholds_DD, edge_densities_DD = genEdgeDensity(ar_nonzero_DD_trans, 100)

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.set_xlabel('threshold')
ax.set_ylabel('Transaction density')
ax.set_title('Direct Requirement', fontsize='large')
ax.plot(theta_thresholds_DR, edge_densities_DR, color='blue', lw=2)
ax = fig.add_subplot(2,1,2)
ax.set_xlabel('threshold')
ax.set_ylabel('Transaction density')
ax.set_title('Direct Demand', fontsize='large')
ax.plot(theta_thresholds_DD, edge_densities_DD, color='blue', lw=2)
fig.tight_layout()
#fig.savefig(ROOTPATH + '/Stock_DWCN/Transaction_density_0.pdf', bbox_inches='tight')
            
df_combined_thresholds = combineThresholds(
    theta_thresholds_DR,
    theta_thresholds_DD,
    EIO_direct_requirements_matrix,
    EIO_direct_demands_matrix)

pt = df_combined_thresholds.pivot_table(index='theta_DR', columns='theta_DD', values='no_directions', aggfunc=np.sum)

f, ax = plt.subplots(figsize = (10, 4))
#cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(pt.iloc[:20,:20], cmap='rainbow', linewidths = 0.05, ax = ax)
ax.set_title('Amounts of directions per DR-threshold and DD-threshold')
ax.set_xlabel('theta_DD')
ax.set_ylabel('theta_DR')
#f.savefig(ROOTPATH + '/Stock_DWCN/sns_heatmap_theta_small.pdf', bbox_inches='tight')

###------------ -------------------------- ------------###

###------------ Dealing with Graphs ------------###
### The following graphs are generated here:
### PGD: Pure Dedirected Graph without weights
### UGD: Directed Graph with 'weight' as economical
###      ..cash transactions and 'corr' as abnormal
###      ..return correlations
### PCGD: Directed Graph with weight as partial abnormal
###      ..return correlations

# function for generating PGD
def genPureDedirectedGraph(theta_1, theta_2, mat_1, mat_2):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for i in lst_tickers_stp:
        print(i, end=' ')
        matidx_i = EIO_industry_BEA_code_list.index(
            df_codes_and_title.loc[i,'BEA'])
        for j in lst_tickers_stp:
            if i is not j:
                matidx_j = EIO_industry_BEA_code_list.index(
                    df_codes_and_title.loc[j,'BEA'])
                a = mat_1[matidx_i, matidx_j]
                b = mat_2[matidx_i, matidx_j]
                if a > 0.0 and logarise(a) > theta_1:
                    G.add_edge(i, j)
                if b > 0.0 and logarise(b) > theta_2:
                    G.add_edge(j, i)
    return G

# function for generating UGD
def genUniversalGraph(stock_return_df):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for i in lst_tickers_stp:
        matidx_i = EIO_industry_BEA_code_list.index(df_codes_and_title.loc[i,'BEA'])
        S1 = stock_return_df[i]
        for j in lst_tickers_stp:
            matidx_j = EIO_industry_BEA_code_list.index(df_codes_and_title.loc[j,'BEA'])
            S2 = stock_return_df[j]
            if i is not j:
                if EIO_direct_requirements_matrix[matidx_i, matidx_j] > 0:
                    G.add_edge(i, j)
                    G[i][j]['direct_requirement'] = EIO_direct_requirements_matrix[matidx_i, matidx_j]
                if EIO_direct_demands_matrix[matidx_j, matidx_i] > 0:
                    G.add_edge(i, j)
                    G[i][j]['direct_demand'] = EIO_direct_demands_matrix[matidx_j, matidx_i]
                if G.has_edge(i, j):
                    G.add_edge(i, j, corr=S1.corr(S2))
    return G

def genWDGraphFromPureDirectedGraph(PGD, return_list, threshold=-1):
    G = nx.DiGraph()
    G.add_nodes_from(lst_tickers_stp)
    for n, nbrs in PGD.adj.items():
        S1 = return_list[n]
        for nbr, eattr in nbrs.items():
            S2 = return_list[nbr]
            corr = S1.corr(S2)
            if corr >= threshold:
                G.add_edge(n, nbr, corr=corr)
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

PGD = genPureDedirectedGraph(
    0.35,
    0.33,
    EIO_direct_requirements_matrix,
    EIO_direct_demands_matrix)

WGD_NR = genWDGraphFromPureDirectedGraph(PGD, df_stock_normal_return, 0.4)

WDGD = genWDGraph(PGD, 0.2)

UGD = genUniversalGraph()
PCGD = genPartialCorrGraph(UGD)
ECGU = genEntCorrGraph()

for n, nbrs in DG.adj.items():
    print(n)
    for nbr, eattr in nbrs.items():
        EDG.add_edge(n, nbr, weight=eattr['corr'])

DWG = nx.DiGraph()
DWG.add_nodes_from(lst_tickers_stp)

###------- properties ------###
corr_coef_CN = []
for n, nbrs in UGD.adj.items():
    print(n, end=' ')
    for nbr, eattr in nbrs.items():
        corr_coef_CN.append(eattr['corr'])

direct_requirements_CN = []
direct_demands_CN = []
a = None
for n, nbrs in UGD.adj.items():
    print(n, end=' ')
    for nbr, eattr in nbrs.items():
        if 'direct_requirement' in eattr.keys():
            direct_requirements_CN.append(eattr['direct_requirement'])
        if 'direct_demand' in eattr.keys():
            direct_demands_CN.append(eattr['direct_demand'])

## Distribution of correlation coefficients
plt.hist(corr_coef_CN, density=1, bins=200, histtype='bar')
plt.axis([-0.35, 1, 0, 3.1])
#axis([xmin,xmax,ymin,ymax])
plt.xlabel('correlation')
plt.ylabel('p(correlation)')
plt.title('correlation coefficient distribution')
#plt.savefig(ROOTPATH + '/Stock_DWCN/correlation_coefficient_distribution_0.pdf', bbox_inches='tight')

describe(corr_coef_CN)

theta_thresholds_corr, edge_densities_corr = genEdgeDensity(corr_coef_CN, 100)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(theta_thresholds_corr, edge_densities_corr, color='blue', lw=2)
plt.xlabel('correlation coefficient threshold')
plt.ylabel('edge density')
plt.title('stock correlation network edge density')
#fig.savefig(ROOTPATH + '/Stock_DWCN/stock_correlation_network_edge_density_0.pdf', bbox_inches='tight')

###------------ Unweighted Directed Netwrok Analysis ------------###
### Frequency distributions
def plotDegreeDistributeion(G, density=1, degree_type='overall', no_bins=50, save_file=None):
    if degree_type is 'overall':
        degree = G.degree()
    elif degree_type is 'in':
        degree = G.in_degree()
    elif degree_type is 'out':
        degree = G.out_degree()
    lst_degree = [i[1] for i in degree]
    plt.hist(lst_degree, density=density, bins=no_bins, histtype='bar')
    #plt.axis([-0.7, 1, 0, 4.1])
    #axis([xmin,xmax,ymin,ymax])
    plt.xlabel('correlation')
    plt.ylabel('p(correlation)')
    plt.title('correlation coefficient distribution')
    if save_file is not None:
        plt.savefig(
            ROOTPATH +
            '/Stock_DWCN/%s_degree_distribution_distribution_%s.pdf' %
            (degree_type, save_file),
            bbox_inches='tight')

def getRangeDegreeCountOfIndustries(degree_view, theta_low_mid, theta_mid_high):
    df = pd.DataFrame(index=np.unique(df_codes_and_title['NAICS']),
                      columns=[
                          'Entire', 'Zero Degree',
                          'Low Degree', 'Mid Degree',
                          'High Degree'], data=0)
    df.index.name = 'Amount of degrees'
    lst_value_counts = [pd.DataFrame()] * 5
    lst_value_counts[0] = pd.Series(getIndustryCodeByStockCode(
        [i[0] for i in degree_view],
        code_type='Title')).value_counts()
    lst_value_counts[1] = pd.Series(getIndustryCodeByStockCode(
        [i[0] for i in degree_view if i[1] == 0],
        code_type='Title')).value_counts()
    lst_value_counts[2] = pd.Series(getIndustryCodeByStockCode(
        [i[0] for i in degree_view if i[1] > 0 and i[1] < theta_low_mid],
        code_type='Title')).value_counts()
    lst_value_counts[3] = pd.Series(getIndustryCodeByStockCode(
        [i[0] for i in degree_view if i[1] >= theta_low_mid and i[1] < theta_mid_high],
        code_type='Title')).value_counts()
    lst_value_counts[4] = pd.Series(getIndustryCodeByStockCode(
        [i[0] for i in degree_view if i[1] >= theta_mid_high],
        code_type='Title')).value_counts()
    for i in range(len(lst_value_counts)):
        value_counts = lst_value_counts[i]
        for j in value_counts:
            df.ix[j[0], i] = j[1]
    return df

overall_degree_WGD_NR = WGD_NR.degree()
in_degree_WGD_NR = WGD_NR.in_degree()
out_degree_WGD_NR = WGD_NR.out_degree()
[print(i[0]) for i in out_degree if i[1] == sorted(lst_out_degree, reverse=True)[0]]

df_range_degree_count_of_industries = getRangeDegreeCountOfIndustries(overall_degree_WGD_NR, 200, 1000)
# Plot stacked bar chart

### Path length
shortest_path = nx.shortest_path(PGD)
average_shortest_path_length = nx.average_shortest_path_length(PGD) # 1.1990808799776556


###------------ ------------------------------------ ------------###

###------------ Weighted Directed Netwrok Analysis ------------###

###------------ ---------------------------------- ------------###
# threshold-EIO | threshold-corr | nonodes, noedges | rmved | glbEffc, locEffc | avg_short_path | avg_clustering
# 0.76| 0.36 | 1418, 82968 | 1247, 82968 | 0.1486061381592153, 0.7465691741625827 | 1.432 | 0.5791134189676599



'''
corr_coef_CN_DiG = []
for n, nbrs in DiUnwtG.adj.items():
    S1 = df_stock_normal_return[n]
    for nbr, _ in nbrs.items():
        S2 = df_stock_normal_return[nbr]
        corr_coef_CN_DiG.append(S1.corr(S2))

plt.hist(corr_coef_CN_DiG, density=1, bins=200, histtype='bar')
plt.axis([-0.35, 1, 0, 3.1])
#axis([xmin,xmax,ymin,ymax])
plt.xlabel('correlation')
plt.ylabel('p(correlation)')
plt.title('correlation coefficient distribution')

complete_corr_coef_CN = [None] * LENTRGL
cnt = 0
for i in range(LENTCKR):
    S1 = df_stock_normal_return[lst_tickers_stp[i]]
    for j in range(i+1, LENTCKR):
        S2 = df_stock_normal_return[lst_tickers_stp[j]]
        complete_corr_coef_CN[cnt] = S1.corr(S2)
        cnt += 1
        
plt.hist(complete_corr_coef_CN, density=1, bins=200, histtype='bar')
plt.axis([-0.35, 1, 0, 3.1])
#axis([xmin,xmax,ymin,ymax])
plt.xlabel('correlation')
plt.ylabel('p(correlation)')
plt.title('correlation coefficient distribution')

df_combined_thresholds = combineThresholds(
    theta_thresholds_DR,
    theta_thresholds_DD,
    EIO_direct_requirements_matrix,
    EIO_direct_demands_matrix)
    
pt = df_combined_thresholds.pivot_table(index='theta_DR', columns='theta_DD', values='no_directions', aggfunc=np.sum)

f, ax = plt.subplots(figsize = (10, 4))
#cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(pt.iloc[:20,:20], cmap='rainbow', linewidths = 0.05, ax = ax)
ax.set_title('Amounts of directions per DR-threshold and DD-threshold')
ax.set_xlabel('theta_DD')
ax.set_ylabel('theta_DR')

degree_sequence = sorted([d for n, d in DiWtG_NmRt_1.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

mat_max = 0
for m in EIO_direct_demands_matrix:
    max_t = max(np.array(m)[0])
    if max_t > mat_max and max_t < 1:
        mat_max = max_t
        print('mat_max: %s' % mat_max)
for i in range(EIO_direct_demands_matrix.shape[0]):
    ar_m = np.array(EIO_direct_demands_matrix[i])[0]
    for j in range(len(ar_m)):
        if ar_m[j] >= 1:
            EIO_direct_demands_matrix[i, j] = np.ceil(mat_max*10.0)/10.0
            print('EIO_direct_demands_matrix[%d, %d]: %s' % (i, j, EIO_direct_demands_matrix[i, j]))
            
def genSmallWorldGraph(nonodes, noedges, G_sw):
    print('Converting to directed graph...')
    DG_sw = G_sw.to_directed()
    lst_edge = list(DG_sw.edges())
    print('Triming...')
    lst_idx = np.random.choice(len(lst_edge), size=DG_sw.number_of_edges()-noedges, replace=False)
    for i in lst_idx: DG_sw.remove_edge(lst_edge[i][0], lst_edge[i][1])
    return DG_sw
    
G_sw = nx.generators.watts_strogatz_graph(n=nonodes, k=nonodes-1, p=0.5)
DG_sw = genSmallWorldGraph(nonodes, noedges, G_sw)


degree_sequence = sorted([d for n, d in DiG_connected.degree()])
degree_count = collections.Counter(degree_sequence)
deg, cnt = zip(*degree_count.items())

strength_sequence = sorted([s for n, s in G.degree(weight='corr') if s > 0])
strength_count = collections.Counter(degree_sequence)
stren, stren_cnt = zip(*degree_count.items())

%%time
e_c = nx.eigenvector_centrality(G)
sum(e_c.values())
sum(sorted([e_c[e] for e in e_c.keys()],reverse=True))


'''
