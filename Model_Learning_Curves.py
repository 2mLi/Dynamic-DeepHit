# this script produces the training learning curves of different models
# two plots will be generated: c-index and total loss trend


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

# argv[1] is the data_mode: eg if PreCar, the program will read it from the PreCar file
# argv[2], if left empty, will choose the most recent log
# if argv[2] is specified, will use the string to find relevant log

data_mode_name = sys.argv[1]

if len(sys.argv) < 3: 
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

# extract train results

train_results = params['train_results']

val_c_idx = train_results['val_c_idx']
c_track = train_results['c_track']
c_track_improve_idx = train_results['c_track_improve_idx']
total_loss = train_results['total_loss']

import matplotlib.pyplot as plt

eval_path = target_dir + '/learning_curves'
if not os.path.exists(eval_path):
    os.makedirs(eval_path)



# for loss curve, need how many burnins and how many keeps
burnin = new_parser['iteration_burn_in']
keep = new_parser['iteration']
# the first loss corresponds to the first burnin + 1000
l = len(total_loss)
x_loss = [(i + 1) * 1 + burnin for i in range(l)]
ub_total_loss = float(np.quantile(total_loss, 0.975))
lb_total_loss = float(np.quantile(total_loss, 0.025))

# for c-index we wanna show the process of plateuing... 
target = 1
c_index = [0.5]
for idx in x_loss: 
    if idx - burnin < c_track_improve_idx[target]: 
        c_index.append(c_track[target - 1])
    else: 
        c_index.append(c_track[target])
        target = target + 1
        if target + 1 > len(c_track): 
            break; 


# c-index plot
c_track_improve_idx = [i + burnin for i in c_track_improve_idx]
plt.figure()
plt.axvline(burnin, 0, 1, linestyle = '--', color = 'black', linewidth= 0.75)
# plt.plot(c_track_improve_idx, c_track, linestyle = '--', color = 'black')
# plt.plot(x_loss, c_index)
# plt.step(c_track_improve_idx, c_track, where = 'pre', color = 'red')
plt.step(c_track_improve_idx, c_track, where = 'post', color = 'cyan')
plt.xlabel('Iteration')
plt.ylabel('c-index')


plt.savefig(eval_path + "/c_index.png")

plt.figure()
plt.xlim((burnin, max(x_loss) + 1000))
plt.plot(x_loss, total_loss, linewidth = 0.5, color = 'cyan')
# maybe... we can add two lines that represent the 95% CI in this region

plt.axhline(lb_total_loss, 0, 1, linestyle = '--', color = 'black', linewidth = 0.75)
plt.axhline(ub_total_loss, 0, 1, linestyle = '--', color = 'black', linewidth = 0.75)
plt.xlabel('Iteration')
plt.ylabel('Total loss')
plt.savefig(eval_path + "/total_loss.png")