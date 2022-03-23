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

# argv[1] is the data_mode: eg if PreCar, the program will read it from the PreCar file
# argv[2], if left empty, will choose the most recent log
# if argv[2] is specified, will use the string to find relevant log

data_mode_name = sys.argv[1]

if len(sys.argv) < 6: 
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

if len(sys.argv) < 6: 
    # this means no argv[2] is given; we use the most recent log
    # then, new eval and pred time would be argument argv[2] and argv[3]
    eval_time = float(sys.argv[2])
    pred_time = float(sys.argv[3])
    step = float(sys.argv[4])
    pat_idx = int(sys.argv[5]) - 1
else: 
    eval_time = float(sys.argv[3])
    pred_time = float(sys.argv[4])
    step = float(sys.argv[5])
    pat_idx = int(sys.argv[6]) - 1
# for this patient... (in test set)
# here, assume the patient comes from test set 1
te_data_s = te_data[pat_idx, :, :]
te_data_s = te_data_s[newaxis, :, :]
te_data_mi_s = te_data[pat_idx, :, :]
te_data_mi_s = te_data_mi_s[newaxis, :, :]

# modify the true pred_time
steps = round(pred_time/step)
true_eval_time = [step * i for i in range(steps)]
true_eval_time.append(pred_time)



risk = f_get_risk_predictions(sess, model, te_data_s, te_data_mi_s, [pred_time], true_eval_time)
risk = risk[0][0, 0, :]

# remove float32 tag
risk = [float(i) for i in risk]

t = [i + eval_time for i in true_eval_time]
t = [float(i) for i in t]

# this would be a string of risks
# also export the longitudinal trajectory
# to do so...
export_file = {
    "patient_idx": int(pat_idx + 1), 
    "t": t, 
    "risk": risk
}

long_traj_dir = target_dir + '/eval/pat_long_traj/'

if not os.path.exists(long_traj_dir):
    os.makedirs(long_traj_dir)

landmark_horizon_lab = 'L' + str(int(eval_time)) + 'H' + str(int(pred_time))

with open(long_traj_dir + landmark_horizon_lab + '_log.json', "w") as f:
    json.dump(export_file, f)

