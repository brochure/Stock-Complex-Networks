import sys, csv, time, requests, statsmodels, math
from datetime import datetime, timedelta
from core import LocateIdx, Polygonal, Triangular, SendEmail, ROOTPATH, DROPLIST, df_source
from preprocess_stocks import lst_tickers_stp, LENTCKR, LENTRGL
from sampling_granger import lst_rdm_pair, srs_res_sample
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.base import clone
import networkx as nx
import pandas as pd
import numpy as np

skfolds = StratifiedKFold(n_splits=3, random_state=42)
kf = KFold(n_splits=3, shuffle = True)

df_sample = pd.DataFrame(columns=df_source.columns)
for rdm_pair in lst_rdm_pair:
    idx = LocateIdx(rdm_pair[0], rdm_pair[1], num_of_obj=LENTCKR)
    df_sample = df_sample.append(df_source.iloc[idx,:])
df_sample = df_sample.reset_index(drop=True)
for i in range(len(srs_res_sample)):
    if srs_res_sample[i] == -1:
        srs_res_sample[i] = 2
df_sample['causality'] = srs_res_sample
df_sample['corr_lbl'] = 0
cnt = 0
sri_corr = df_sample['corr']
for i in range(len(sri_corr)):
    if sri_corr[i] > 0.8:
        df_sample.loc[i,'corr_lbl'] = 1
    elif sri_corr[i] > 0.5:
        df_sample.loc[i,'corr_lbl'] = 1
#dnn_rgs.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

corr_lbl = tf.feature_column.numeric_column('corr_lbl')
#coint_pvalue = tf.feature_column.numeric_column('coint_pvalue')
ieas = tf.feature_column.numeric_column('ieas')
pricetorevenue = tf.feature_column.numeric_column('pricetorevenue')
marketcap = tf.feature_column.numeric_column('marketcap')
pricetobook = tf.feature_column.numeric_column('pricetobook')
pricetoearnings = tf.feature_column.numeric_column('pricetoearnings')
beta = tf.feature_column.numeric_column('beta')


model_dir = ROOTPATH + r'/Stock_DWCN/ml_model_data/deep_0616_01.csv'
deep_columns = [
    #coint_pvalue,
    ieas,
    #pricetorevenue,
    #marketcap,
    pricetobook,
    #pricetoearnings,
    beta
]

estimator = tf.estimator.DNNRegressor(
    model_dir=model_dir,
    feature_columns=deep_columns,
    hidden_units=[10,10],
    n_classes=3
)

col_indpd = ['ieas', 'pricetorevenue', 'marketcap', 'pricetobook', 'pricetoearnings', 'beta']
col_tgdpd = 'corr' #causality

df_sample_sfl = shuffle(df_sample).reset_index(drop=True)
# balancing according to each label
arr_uniq_lbl = np.unique(df_sample['causality'])
lst_res_sample = list(df_sample['causality'])
lst_uniq_lbl_cnt = []
for i in range(len(arr_uniq_lbl)):
    lst_uniq_lbl_cnt.append(lst_res_sample.count(i))
for i in range(len(lst_uniq_lbl_cnt)):
    for j in range(lst_uniq_lbl_cnt[i]-min(lst_uniq_lbl_cnt)):
        df_sample_sfl = df_sample_sfl.drop(df_sample_sfl[df_sample_sfl['causality']==i].iloc[0].name)
    
df_sample_sfl = shuffle(df_sample_sfl).reset_index(drop=True)
df_sample_sfl_a = df_sample_sfl.loc[:2403,:].reset_index(drop=True)
df_sample_sfl_b = df_sample_sfl.loc[2403:,:].reset_index(drop=True)
X_train = df_sample_sfl_a.loc[:, col_indpd].reset_index(drop=True)
y_train = df_sample_sfl_a.loc[:, col_tgdpd].reset_index(drop=True)
#df_source_sfl_2 = df_source_sfl.iloc[len(df_source_sfl)/2:(len(df_source_sfl)/2+2000)].reset_index()
X_eval = df_sample_sfl_b.loc[:, col_indpd].reset_index(drop=True)
y_eval = df_sample_sfl_b.loc[:, col_tgdpd].reset_index(drop=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn({
        'ieas': X_train['ieas'],
        'pricetorevenue': X_train['pricetorevenue'],
        'marketcap': X_train['marketcap'],
        'pricetobook': X_train['pricetobook'],
        'pricetoearnings': X_train['pricetoearnings'],
        'beta': X_train['beta'],
    },
    y_train, batch_size=50,
    num_epochs=None, shuffle=True
)

test_input_fn = tf.estimator.inputs.numpy_input_fn({
        'ieas': X_train['ieas'],
        'pricetorevenue': X_train['pricetorevenue'],
        'marketcap': X_train['marketcap'],
        'pricetobook': X_train['pricetobook'],
        'pricetoearnings': X_train['pricetoearnings'],
        'beta': X_train['beta'],
    },
    y_train, batch_size=50,
    num_epochs=40000, shuffle=False
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn({
        'ieas': X_eval['ieas'],
        'pricetorevenue': X_eval['pricetorevenue'],
        'marketcap': X_eval['marketcap'],
        'pricetobook': X_eval['pricetobook'],
        'pricetoearnings': X_eval['pricetoearnings'],
        'beta': X_eval['beta'],
    },
    y_eval, batch_size=50,
    num_epochs=40000, shuffle=False
)

estimator.train(input_fn=train_input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=test_input_fn, name='training_data')
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, name='test_data')

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

df_source_sfl_3 = df_source_sfl.iloc[int(len(df_source_sfl)/2)+10001:int(len(df_source_sfl)/2+10050)].reset_index()
X_pred = df_source_sfl_3.loc[:, ['ieas', 'coint_pvalue']]
y_pred = df_source_sfl_3.loc[:, 'corr']

pred_input_fn = tf.estimator.inputs.numpy_input_fn({
        'ieas': X_pred['ieas'],
        'pricetorevenue': X_pred['pricetorevenue'],
        'marketcap': X_pred['marketcap'],
        'pricetobook': X_pred['pricetobook'],
        'pricetoearnings': X_pred['pricetoearnings'],
        'beta': X_pred['beta'],
    },
    num_epochs=40000, shuffle=False
)

a=estimator.predict(input_fn=pred_input_fn)

