# okay, now combine everything together...

_EPSILON = 1e-08

#### <<< Warning suppression>>> ###
# import warnings
# warnings.filterwarnings('deprecated')
#### This makes the resulting log a lot nicer BUT could produce errors in very, very rare and unexpected circumstances. 

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import sys
import json
import time as timepackage

from numpy import newaxis

from sklearn.model_selection import train_test_split

import import_data as impt

from class_DeepLongitudinal import Model_Longitudinal_Attention

from utils_eval             import c_index, brier_score
from utils_log              import save_logging, load_logging
from utils_helper           import f_get_minibatch, f_get_boosted_trainset



def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    """
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    """
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :]    = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi)


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time):
    
    pred = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event, num_Category = np.shape(pred)
       
    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros([np.shape(data_)[0], len(pred_time), len(eval_time)])
            
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)


        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon #if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,pred_horizon:(eval_horizon+1)], axis=2) #risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:,:,pred_horizon:], axis=2), axis=1, keepdims=True) +_EPSILON) #conditioniong on t > t_pred
            
            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]
                
    return risk_all

## cmd args: 
# now only one argument is needed
# this will be something like "PreCar"
# and the machine will know to find all relevant materials from the "PreCar" directory





### the following codes read model training results plus needed data from Model_Training.py
# and theoretically can be used to re-construct everything needed? 

'''
saver.restore(sess, sys.argv[1])
with open(sys.argv[2]) as p: 
    params = json.load(p)
'''

# sys.argv = ['mod', 'PreCar', '6', '0.01', '1', '2313', 'LEFT']

# argv[1] is the data_mode: eg if PreCar, the program will read it from the PreCar file
# argv[2], if left empty, will choose the most recent log
# if argv[2] is specified, will use the string to find relevant log

data_mode_name = sys.argv[1]

if len(sys.argv) < 7:  
    # this means no argv[2] is given; we use the most recent log
    # to do so, for now lets just use max argument
    # firstly, take out all log.json documents
    logs = os.listdir(data_mode_name)
    # logs is a list of all available logs; find the most recent one...
    target_dir = data_mode_name + '/' + max(logs)
    print('Using the most recent _log.json by default, since no specification is given. ')
else: 
    # assume that argv[2] has specified a keyword, use the keyword to identify logs
    logs = os.listdir(data_mode_name)
    matched = [i for i in logs if sys.argv[2] in i]
    if len(matched) >= 2: 
        print('Warning: more than one log is matched with the keyword and the most recent one will be used. ')
        matched = max(matched)
    target_dir = data_mode_name + '/' + matched[0]


# read log
with open(target_dir + '/' + '_log.json') as p: 
    params = json.load(p)
mod_dir = target_dir + '/' + 'model'

# print(type(params))
new_parser = params['new_parser']
dataset_info = params['dataset_info']
evaluation_info = params['evaluation_info']
model_configs = params['model_configs']
eval_configs = params['eval_configs']
time_tag = params['new_parser']['time_tag']

dirs = dataset_info
test_dir = []
data_mode = data_mode_name
for key in list(dirs.keys()): 
    if key == data_mode: 
        train_dir = dirs[key]
    else: 
        test_dir.append(dirs[key])

(tr_x_dim, tr_x_dim_cont, tr_x_dim_bin), (tr_data, tr_time, tr_label), (tr_mask1, tr_mask2, tr_mask3), (tr_data_mi), (tr_id), tr_feat_list = impt.import_dataset(path = train_dir, bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

(te_x_dim, te_x_dim_cont, te_x_dim_bin), (te_data, te_time, te_label), (te_mask1, te_mask2, te_mask3), (te_data_mi), (te_id), te_feat_list = impt.import_dataset(path = test_dir[0], bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

(tea_x_dim, tea_x_dim_cont, tea_x_dim_bin), (tea_data, tea_time, tea_label), (tea_mask1, tea_mask2, tea_mask3), (tea_data_mi), (tea_id), tea_feat_list = impt.import_dataset(path = test_dir[1], bin_list_in = model_configs['bin_list'], cont_list_in = model_configs['cont_list'], log_list = model_configs['log_transform'])

if tr_data.shape[1] > te_data.shape[1] : 
    # this means te_data have fewer follow-ups than tr_data. For this, patch it up with vectors of zero. 
    print('Test set [1] has fewer follow-ups than train set. Artificially generated follow-ups have been attached. ')
    k = tr_data.shape[1] - te_data.shape[1]
    for i in range(k): 
        te_data = np.append(te_data, np.zeros(shape = (te_data.shape[0], 1, te_data.shape[2]), dtype = float), axis = 1) 
        te_data_mi = np.append(te_data_mi, np.zeros(shape = (te_data_mi.shape[0], 1, te_data_mi.shape[2]), dtype = float), axis = 1) 

if tr_data.shape[1] > tea_data.shape[1] : 
    
    print('Test set [2] has fewer follow-ups than train set. Artificially generated follow-ups have been attached. ')
    k = tr_data.shape[1] - tea_data.shape[1]
    for i in range(k): 
        tea_data = np.append(tea_data, np.zeros(shape = (tea_data.shape[0], 1, tea_data.shape[2]), dtype = float), axis = 1) 
        tea_data_mi = np.append(tea_data_mi, np.zeros(shape = (tea_data_mi.shape[0], 1, tea_data_mi.shape[2]), dtype = float), axis = 1) 

# on the other hand what may happen if... 
if tr_data.shape[1] < te_data.shape[1] : 
    # this means te_data have fewer follow-ups than tr_data. For this, patch it up with vectors of zero. 
    print('Test set [1] has fewer follow-ups than train set. Artificially curtailed excessive follow-ups to avoid critical failures. ')
    te_data = te_data[:, range(tr_data.shape[1]), :]
    te_data_mi = te_data_mi[:, range(tr_data_mi.shape[1]), :]

if tr_data.shape[1] < tea_data.shape[1] : 
    
    print('Test set [2] has fewer follow-ups than train set. Artificially curtailed excessive follow-ups to avoid critical failures. ')
    tea_data = tea_data[:, range(tr_data.shape[1]), :]
    tea_data_mi = tea_data_mi[:, range(tr_data_mi.shape[1]), :]

pred_time = evaluation_info['pred_time'] # prediction time (in months)
eval_time = evaluation_info['eval_time'] # months evaluation time (for C-index and Brier-Score)

_, num_Event, num_Category  = np.shape(tr_mask1)  # dim of mask3: [subj, Num_Event, Num_Category]

max_length                  = np.shape(tr_data)[1]

#####

# A little treat: print name (in dict) of dataset
def get_key(val):
    for key, value in dataset_info.items():
         if val == value:
             return key
 
    return "There is no such Key"

train_name = get_key(train_dir)
test1_name = get_key(test_dir[0])
test2_name = get_key(test_dir[1])


#####

input_dims                  = { 'x_dim'         : tr_x_dim,
                                'x_dim_cont'    : tr_x_dim_cont,
                                'x_dim_bin'     : tr_x_dim_bin,
                                'num_Event'     : num_Event,
                                'num_Category'  : num_Category,
                                'max_length'    : max_length }

network_settings            = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
                                'h_dim_FC'          : new_parser['h_dim_FC'],
                                'num_layers_RNN'    : new_parser['num_layers_RNN'],
                                'num_layers_ATT'    : new_parser['num_layers_ATT'],
                                'num_layers_CS'     : new_parser['num_layers_CS'],
                                'RNN_type'          : new_parser['RNN_type'],
                                'FC_active_fn'      : tf.nn.relu,
                                'RNN_active_fn'     : tf.nn.tanh,
                                'initial_W'         : tf.contrib.layers.xavier_initializer(),

                                'reg_W'             : new_parser['reg_W'],
                                'reg_W_out'         : float(new_parser['reg_W_out'])
                                 }

tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_Longitudinal_Attention(sess, "Dynamic-DeepHit", input_dims, network_settings)

saver = tf.train.Saver()
saver.restore(sess, mod_dir)

# By default, at each landmark time and horizon, both c-index and Brier score will be computed
# Results will be printed, and saved in a _log.txt document

# here, we superseded eval_time and pred_time: 



if len(sys.argv) < 7: 
    # this means no argv[2] is given; we use the most recent log
    # then, new eval and pred time would be argument argv[2] and argv[3]
    # eval_time = float(sys.argv[2])
    pred_time = float(sys.argv[2])
    step = float(sys.argv[3])
    pat1 = int(sys.argv[4])# {Left or Right}
    grp = int(sys.argv[5])
else: 
    # eval_time = float(sys.argv[3])
    pred_time = float(sys.argv[3])
    step = float(sys.argv[4])
    pat1 = int(sys.argv[5])
    grp = int(sys.argv[6])
# for this patient... (in test set)
# determine which set this is
if grp == 1:  
    te_id = list(te_id)
    te_data = te_data
    te_data_mi = te_data_mi
    idf = test1_name
elif grp == 2: 
    te_id = list(tea_id)
    te_data = tea_data
    te_data_mi = tea_data_mi
    idf = test2_name
elif grp == 0: 
    te_id = list(tr_id)
    te_data = tr_data
    te_data_mi = tr_data_mi
    idf = train_name
else: 
    print("The user has not correctly specified which dataset the patient comes from. Assuming from test set 1. ")
    te_id = list(te_id)
    te_data = te_data
    te_data_mi = te_data_mi
    idf = test1_name

# find pat_idx
if pat1 in te_id: 
    pat1_idx = te_id.index(pat1)
else: 
    print("The specified patient id was not found in the specified test set. Assuming the use of first patient in test set. ")
    pat_idx = 0
    


pat1_data = te_data[pat1_idx, :, :]
pat1_data = pat1_data[newaxis, :, :]
pat1_data_mi = te_data_mi[pat1_idx, :, :]
pat1_data_mi = pat1_data_mi[newaxis, :, :]


# work out the true eval time
# the first element is always zero...
true_eval_time1 = [0]
pat_time_series1 = pat1_data[0, :, 0]
pat_time_series1 = [i for i in pat_time_series1 if not i == 0]

for i in range(len(pat_time_series1)): 
    true_eval_time1.append(pat_time_series1[i]) # append time
    
l1 = len(pat_time_series1)
for i in [int(j) for j in list(np.linspace(1, l1, l1))]: 
    true_eval_time1[i] = true_eval_time1[i] + true_eval_time1[i - 1]

# for pred_time, let us still use the external argument
# pred_time = float(sys.argv[3])
steps = round(pred_time/step)
true_pred_time = [step * i for i in range(steps)]
true_pred_time.append(pred_time)

# finally, risks
risk1 = f_get_risk_predictions(sess, model, pat1_data, pat1_data_mi, true_eval_time1, true_pred_time)
risk1 = risk1[0]
# print(str(true_eval_time1))

# plotting

import math

import matplotlib.pyplot as plt

# first, the longitudinal data
# first, extract continuous biomarkers
cont_list = model_configs['cont_list']

# extract x dim info

x_dim_cont = input_dims['x_dim_cont']
cont_range = range(1, 1 + x_dim_cont)
long_data_to_plot1 = pat1_data[0, :, cont_range]


# does this patient become HCC? 
if te_label[pat1_idx, ] == 1: 
    print('Patient Status: HCC')
else: 
    print('Patient Status: LC')


xMAX = max([max(true_eval_time1)])

# consider conditions where x_dim_cont == 0, 1 or > 1

fig_dir = target_dir + '/eval/patTraj/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

fig_name = 'tv_risk_' +  idf + '_Patient_' + str(pat1) + '_horizon_' + str(pred_time) + '_longitudinal_trajectory.jpg'

if x_dim_cont <= 0: 
    print('Warning: No continuous variable detected. Therefore no continuous variable longitudinal trajectory would be plotted. ')
elif x_dim_cont == 1: 
    print('Only one longitudinal variable detected. ')
    x1_plot = true_eval_time1
        # print(str(long_data_to_plot1.shape))
    data_to_plot1_sub = list(long_data_to_plot1[0, range(len(x1_plot))])
    x1_plot = [i for i in x1_plot]
    data_to_plot1_sub = [i for i in data_to_plot1_sub]
    plt.plot(x1_plot, data_to_plot1_sub, 'm.-.')
    plt.xlim((0, xMAX + 2))
    plt.text(0.5, 0, 'Time', ha = 'center')
    plt.savefig(target_dir + '/eval/patTraj/' + fig_name)
else: 
    fig, ax = plt.subplots(x_dim_cont, 1, figsize=(8,6), sharex = True, sharey = True)
    for i in range(x_dim_cont): 
        x1_plot = true_eval_time1
        # print(str(long_data_to_plot1.shape))
        data_to_plot1_sub = list(long_data_to_plot1[i, range(len(x1_plot))])
        # print(str(data_to_plot1_sub))
        
        # print(str(x1_plot))
        # print(str(data_to_plot1_sub))

        x1_plot = [i for i in x1_plot]
        data_to_plot1_sub = [i for i in data_to_plot1_sub]
        ax[i, ].plot(x1_plot, data_to_plot1_sub, 'm.-.', color='blue')
        ax[i, ].set_xlim((0, xMAX + 2))
    fig.text(0.5, 0, 'Time', ha = 'center')
    plt.savefig(target_dir + '/eval/patTraj/' + fig_name)

# plt.ylabel('Predicted risk')
x_plot_sub = []
y_plot_sub = []
for t in range(len(true_eval_time1) - 1): # here minus 1 since we don't want the last follow-up
    x1_plot = [true_eval_time1[t] + i for i in true_pred_time]
    y1_plot = list(risk1[0, t, :])
    # then subset x_plot's part smaller than true_eval_time[1]
    x_plot_sub_temp = [x1_plot[i] for i in range(len(x1_plot)) if x1_plot[i] <= true_eval_time1[t + 1]]
    y_plot_sub_temp = [y1_plot[i] for i in range(len(x1_plot)) if x1_plot[i] <= true_eval_time1[t + 1]]
    # add them to x_plot_sub and y_plot_sub

    x_plot_sub = [x_plot_sub, x_plot_sub_temp]
    x_plot_sub = [item for sublist in x_plot_sub for item in sublist]
    y_plot_sub = [y_plot_sub, y_plot_sub_temp]
    y_plot_sub = [item for sublist in y_plot_sub for item in sublist]
# print(x_plot_sub)
# print(y_plot_sub)
plt.figure(figsize = (8, 8))
plt.xlim((0, xMAX + 2))
plt.ylim((0, max(y_plot_sub) * 1.1))
plt.xlabel('Time')
plt.plot(x_plot_sub, y_plot_sub, 'b')
    # add a vertical line
for t in range(len(true_eval_time1) - 1): 
    plt.vlines(true_eval_time1[t + 1], 0, max(y_plot_sub) * 1.1, 'k', '--')
    # save

fig_dir = target_dir + '/eval/patTraj/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

fig_name = 'tv_risk_' +  idf + '_Patient_' + str(pat1) + '_horizon_' + str(pred_time) + '_risk_trajectory.jpg'
plt.savefig(target_dir + '/eval/patTraj/' + fig_name)