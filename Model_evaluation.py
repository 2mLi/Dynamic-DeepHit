_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
import json
import time as timepackage

from sklearn.model_selection import train_test_split

import import_data as impt

from class_DeepLongitudinal import Model_Longitudinal_Attention

from utils_eval             import c_index, brier_score
from utils_log              import save_logging, load_logging
from utils_helper           import f_get_minibatch, f_get_boosted_trainset

## cmd args: 
# argv[0]: Model_evaluation.py
# argv[1]: address of sess model; must be created by Model_Training.py
# argv[2]: params used to build Model_Training.py
# for now, the user must make sure argv[2] is the one used to construct argv[1]





### the following codes read model training results plus needed data from Model_Training.py
# and theoretically can be used to re-construct everything needed? 

saver.restore(sess, sys.argv[1])
with open(sys.argv[2]) as p: 
    params = json.load(p)
    
    
new_parser = params['new_parser']
dataset_info = params['dataset_info']
evaluation_info = params['evaluation_info']
model_configs = params['model_configs']
eval_configs = params['eval_configs']
time_tag = params['new_parser']['time_tag']

(tr_x_dim, tr_x_dim_cont, tr_x_dim_bin), (tr_data, tr_time, tr_label), (tr_mask1, tr_mask2, tr_mask3), (tr_data_mi), (tr_id), tr_feat_list = impt.import_dataset(path = train_dir, bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

(te_x_dim, te_x_dim_cont, te_x_dim_bin), (te_data, te_time, te_label), (te_mask1, te_mask2, te_mask3), (te_data_mi), (te_id), te_feat_list = impt.import_dataset(path = test_dir[0], bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

(tea_x_dim, tea_x_dim_cont, tea_x_dim_bin), (tea_data, tea_time, tea_label), (tea_mask1, tea_mask2, tea_mask3), (tea_data_mi), (tea_id), tea_feat_list = impt.import_dataset(path = test_dir[1], bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

pred_time = evaluation_info['pred_time'] # prediction time (in months)
eval_time = evaluation_info['eval_time'] # months evaluation time (for C-index and Brier-Score)

_, num_Event, num_Category  = np.shape(tr_mask1)  # dim of mask3: [subj, Num_Event, Num_Category]

max_length                  = np.shape(tr_data)[1]


file_path = '{}'.format(data_mode)

if not os.path.exists(file_path):
    os.makedirs(file_path)

#####

# By default, at each landmark time and horizon, both c-index and Brier score will be computed
# Results will be printed, and saved in a _log.txt document


risk_all = f_get_risk_predictions(sess, model, tr_data, tr_data_mi, pred_time, eval_time)

for p, p_time in enumerate(pred_time):
    pred_horizon = int(p_time)
    result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):                
        eval_horizon = int(t_time) + pred_horizon
        for k in range(num_Event):
            result1[k, t] = c_index(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
            result2[k, t] = brier_score(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
    
    if p == 0:
        final1, final2 = result1, result2
    else:
        final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
col_header = []
for t_time in eval_time:
    col_header.append('eval_time {}'.format(t_time))

# c-index result
df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# brier-score result
df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

### PRINT RESULTS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX: ')
print(df1)
print('--------------------------------------------------------')
print('- BRIER-SCORE: ')
print(df2)
print('========================================================')
            
# Test the Trained Network (in test 1)

risk_all = f_get_risk_predictions(sess, model, te_data, te_data_mi, pred_time, eval_time)

for p, p_time in enumerate(pred_time):
    pred_horizon = int(p_time)
    result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):                
        eval_horizon = int(t_time) + pred_horizon
        for k in range(num_Event):
            result1[k, t] = c_index(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
            result2[k, t] = brier_score(risk_all[k][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
    
    if p == 0:
        final1, final2 = result1, result2
    else:
        final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
col_header = []
for t_time in eval_time:
    col_header.append('eval_time {}'.format(t_time))

# c-index result
df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# brier-score result
df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

### PRINT RESULTS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX: ')
print(df1)
print('--------------------------------------------------------')
print('- BRIER-SCORE: ')
print(df2)
print('========================================================')


# Test the Trained Network (in test 2)

risk_all = f_get_risk_predictions(sess, model, tea_data, tea_data_mi, pred_time, eval_time)

for p, p_time in enumerate(pred_time):
    pred_horizon = int(p_time)
    result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):                
        eval_horizon = int(t_time) + pred_horizon
        for k in range(num_Event):
            result1[k, t] = c_index(risk_all[k][:, p, t], tea_time, (tea_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
            result2[k, t] = brier_score(risk_all[k][:, p, t], tea_time, (tea_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
    
    if p == 0:
        final1, final2 = result1, result2
    else:
        final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
col_header = []
for t_time in eval_time:
    col_header.append('eval_time {}'.format(t_time))

# c-index result
df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# brier-score result
df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

### PRINT RESULTS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX: ')
print(df1)
print('--------------------------------------------------------')
print('- BRIER-SCORE: ')
print(df2)
print('========================================================')