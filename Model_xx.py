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
sys.argv = ['xx.py', 'PreCar', '191288']
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

# check whether the dimension of tr_data, te_data and tea_data match
# check for second dimension... 
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

# then, train the model with different numbers of time points

layers = tr_data.shape[1]
tr_data_ori = tr_data
# for i in list(range(layers)): 
    # subset data
i = 1
seed = 1234
tr_data = tr_data_ori[:, 0:i, :]

# then, training procedures
(tr_data,va_data, tr_data_mi, va_data_mi, tr_time,va_time, tr_label,va_label, 
 tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3) = train_test_split(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, test_size=model_configs["val_ratio"], random_state = seed) 
boost_mode = 'ON'
if boost_mode == "ON":
    tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3 = f_get_boosted_trainset(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3)


# ### 4. Train the Network

# In[ ]:
total_loss = []
c_track = [0.5]
c_track_improve_idx = [0]

##### CREATE DYNAMIC-DEEPFHT NETWORK
tf.reset_default_graph()

config = tf.ConfigProto(device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 200, 
                intra_op_parallelism_threads = 200,
                log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_Longitudinal_Attention(sess, "Dynamic-DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


### TRAINING - BURN-IN
if burn_in_mode == "ON":
    print( "BURN-IN TRAINING ...")
    for itr in range(iteration_burn_in):
        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3)
        DATA = (x_mb, k_mb, t_mb)
        MISSING = (x_mi_mb)

        _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train)

        if (itr+1)%1000 == 0:
            print("itr: {:04d} | loss: {:.4f}".format(itr+1, loss_curr))


### TRAINING - MAIN
print( "MAIN TRAINING ...")
min_valid = float('-inf')
tmp_valid = 0.5

# early stopping
if model_configs["early_stopping"]["mode"] == "ON": 
    stop_round = model_configs["early_stopping"]["stopping_elapse"] # multiplying this by 1000 is at which round we would stop
    C_index_lim = model_configs["early_stopping"]["stopping_c_index"] # 1 by default; >1 values makes no effect; when c index at val set reaches this thr, we stop training
else: 
    stop_round = float(inf)
    C_index_lim = float(inf)
elp = 0

for itr in range(iteration):
        # if condition satisifed, stop training
    if (elp > stop_round or tmp_valid >= C_index_lim): 
        print("Training terminated early because of reaching early-stopping conditions. ")
        break

    x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3)
    DATA = (x_mb, k_mb, t_mb)
    MASK = (m1_mb, m2_mb, m3_mb)
    MISSING = (x_mi_mb)
    PARAMETERS = (alpha, beta, gamma)

    _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)
    total_loss.append(float(loss_curr))

    if (itr+1)%1000 == 0:
        print("itr: {:04d} | loss: {:.4f}".format(itr+1, loss_curr))

    ### VALIDATION  (based on average C-index of our interest)
    if (itr+1)%1000 == 0:        
        risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time)

        for p, p_time in enumerate(pred_time):
            pred_horizon = int(p_time)
            val_result1 = np.zeros([num_Event, len(eval_time)])

            for t, t_time in enumerate(eval_time):                
                eval_horizon = int(t_time) + pred_horizon
                for k in range(num_Event):
                    val_result1[k, t] = c_index(risk_all[k][:, p, t], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

            if p == 0:
                val_final1 = val_result1
            else:
                val_final1 = np.append(val_final1, val_result1, axis=0)

        tmp_valid = np.mean(val_final1)

        if tmp_valid >  min_valid:
            min_valid = tmp_valid
            saver.save(sess, file_path + "/model")
            print( "updated.... average c-index = " + str("%.4f" %(tmp_valid)))
            c_track.append(tmp_valid)
            c_track_improve_idx.append(itr)
            elp = 0
        else: 
            elp = elp + 1


# save c_index and total_loss

for i in c_track: 
    if isinstance(i, np.float32): 
        i = float(i)
for i in c_track_improve_idx: 
    if isinstance(i, np.float32): 
        i = float(i)
for i in total_loss: 
    if isinstance(i, np.float32): 
        i = float(i)
print('With time point number' + str(i + 1))
print('End-point c-index: ' + str(np.max(c_track)))