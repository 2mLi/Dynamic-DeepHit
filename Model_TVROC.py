_EPSILON = 1e-08

#### <<< Warning suppression>>> ###
# import warnings
# warnings.filterwarnings('deprecated')
#### This makes the resulting log a lot nicer BUT could produce errors in very, very rare and unexpected circumstances. 

from ctypes.wintypes import PLARGE_INTEGER
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
# sys.argv = ['mod', 'PreCar', '1', '6', '10000']
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

if len(sys.argv) < 6: 
    # this means no argv[2] is given; we use the most recent log
    # then, new eval and pred time would be argument argv[2] and argv[3]
    eval_time = float(sys.argv[2])
    pred_time = float(sys.argv[3])
    steps = int(sys.argv[4])
else: 
    eval_time = float(sys.argv[3])
    pred_time = float(sys.argv[4])
    steps = int(sys.argv[5])

# for train...
risk = f_get_risk_predictions(sess, model, tr_data, tr_data_mi, [pred_time], [eval_time])
risk = risk[0][:, 0, 0]

# we need: label, time
label = tr_label[:, 0]
time = tr_time[:, 0]
# true label: 
label_tr = label * (time <= pred_time + eval_time)

# we need a discretised scale from min(risk) to max(risk) in Train set
min_risk = min(risk)
max_risk = max(risk)
step = (max_risk - min_risk)/steps #step width
r = [min_risk + step * i for i in range(steps)]
r = r[1:len(r)]


# at each scale, calculate sens and spec
Lsens = []
Lspec = []
LPPV = []
LNPV = []
for ri in r: 
    label_pred = risk >= ri # predicted label
    sens = sum(label_pred * label_tr)/sum(label_tr)
    spec = 1 - sum((1 - label_pred) * (1 - label_tr))/sum(1 - label_tr)
    PPV = sum(label_pred * label_tr)/sum(label_pred)
    NPV = sum((1 - label_pred) * (1 - label_tr))/sum(1 - label_pred)
    Lsens.append(sens)
    Lspec.append(spec)
    LPPV.append(PPV)
    LNPV.append(NPV)

# print(Lsens)
# print(Lspec)

# get AUC with trapezium rule
rL = len(r) - 1
AUCL = []
for i in list(range(rL)): 
    AUCL.append(1/2 * (Lsens[i] + Lsens[i + 1]) * (Lspec[i + 1] - Lspec[i]))
'''


AUC = - sum(AUCL)
AUC_name = str(np.round(AUC, decimals = 4))
print("Time-varying AUC at landmark " + str(eval_time) + " with horizon " + str(pred_time) + ": " + AUC_name)
'''

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import stat_util    #Compute AUC with 95% confidence interval

score, ci_lower, ci_upper, scores = stat_util.score_ci(label_tr, risk, score_fun=roc_auc_score,seed = 142857)


AUC_name = str(np.round(score, decimals = 3))
AUC_UB = str(np.round(ci_upper, decimals = 3))
AUC_LB = str(np.round(ci_lower, decimals = 3))
print("[Train set] Time-varying AUC at landmark " + str(eval_time) + " with horizon " + str(pred_time) + ": " + AUC_name + " (" + AUC_LB + ", " + AUC_UB + ")")

# store results
# firstly, deal with the fucking disgusting float32 stuff
Lspec_to_save = [float(i) for i in Lspec]
Lsens_to_save = [float(i) for i in Lsens]
AUC = score
AUC_to_save = float(AUC)
r_to_save = [float(i) for i in r]
tv_tr_log = {"spec": Lspec_to_save, 
"sens": Lsens_to_save, 
"AUC": AUC_to_save, 
"steps": r_to_save}

# eval_path = target_dir + '/eval'
tvROC_dir = target_dir + '/eval/tvROC/'

if not os.path.exists(tvROC_dir):
    os.makedirs(tvROC_dir)

landmark_horizon_lab = 'L' + str(eval_time) + 'H' + str(pred_time)
with open(tvROC_dir + landmark_horizon_lab + '_log_train.json', "w") as f:
    json.dump(tv_tr_log, f)

# plot bit
Lspec_train = Lspec
Lsens_train = Lsens
AUC_name_train = AUC_name
AUC_UB_train = AUC_UB
AUC_LB_train = AUC_LB
'''
Fig_name = tvROC_dir + landmark_horizon_lab + '_tvROC_train.png'
from matplotlib import pyplot as plt


f = plt.figure()
f.set_figwidth(6)
f.set_figheight(6)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Landmark Time: ' + str(eval_time) + '; Horizon Time: '+ str(pred_time))
plt.text(x = 0.4, y = 0.1, s = "tvAUC: "+ AUC_name + " (" + AUC_LB + ", " + AUC_UB + ")")

plt.plot(Lspec, Lsens)
# plt.show()
plt.savefig(Fig_name)
'''



# for test [1]: 
risk = f_get_risk_predictions(sess, model, te_data, te_data_mi, [pred_time], [eval_time])
risk = risk[0][:, 0, 0]

# we need: label, time
label = te_label[:, 0]
time = te_time[:, 0]
# true label: 
label_te = label * (time <= pred_time + eval_time)

# we need a discretised scale from 0 to 1
min_risk = min(risk)
max_risk = max(risk)
step = (max_risk - min_risk)/steps #step width
r = [min_risk + step * i for i in range(steps)]
r = r[1:len(r)]


# at each scale, calculate sens and spec
Lsens = []
Lspec = []
LPPV = []
LNPV = []
for ri in r: 
    label_pred = risk >= ri # predicted label
    sens = sum(label_pred * label_te)/sum(label_te)
    spec = 1 - sum((1 - label_pred) * (1 - label_te))/sum(1 - label_te)
    PPV = sum(label_pred * label_te)/sum(label_pred)
    NPV = sum((1 - label_pred) * (1 - label_te))/sum(1 - label_pred)
    Lsens.append(sens)
    Lspec.append(spec)
    LPPV.append(PPV)
    LNPV.append(NPV)
# print(Lsens)
# print(Lspec)

# get AUC with trapezium rule
'''
rL = len(r) - 1
AUCL = []
for i in list(range(rL)): 
    AUCL.append(1/2 * (Lsens[i] + Lsens[i + 1]) * (Lspec[i + 1] - Lspec[i]))

AUC = - sum(AUCL)
'''
# here, an alternative using the stat_util.py

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import stat_util    #Compute AUC with 95% confidence interval

score, ci_lower, ci_upper, scores = stat_util.score_ci(label_te, risk, score_fun=roc_auc_score,seed=142857)


AUC_name = str(np.round(score, decimals = 3))
AUC_UB = str(np.round(ci_upper, decimals = 3))
AUC_LB = str(np.round(ci_lower, decimals = 3))
print("[Test set 1] Time-varying AUC at landmark " + str(eval_time) + " with horizon " + str(pred_time) + ": " + AUC_name + " (" + AUC_LB + ", " + AUC_UB + ")")

# store results
# firstly, deal with the fucking disgusting float32 stuff
Lspec_to_save = [float(i) for i in Lspec]
Lsens_to_save = [float(i) for i in Lsens]
AUC = score
AUC_to_save = float(AUC)
r_to_save = [float(i) for i in r]
tv_te_log = {"spec": Lspec_to_save, 
"sens": Lsens_to_save, 
"AUC": AUC_to_save, 
"steps": r_to_save}

# eval_path = target_dir + '/eval'
tvROC_dir = target_dir + '/eval/tvROC/'

if not os.path.exists(tvROC_dir):
    os.makedirs(tvROC_dir)

landmark_horizon_lab = 'L' + str(eval_time) + 'H' + str(pred_time)
with open(tvROC_dir + landmark_horizon_lab + '_log_test_1.json', "w") as f:
    json.dump(tv_te_log, f)

# plot bit
Lspec_test = Lspec
Lsens_test = Lsens
AUC_name_test = AUC_name
AUC_UB_test = AUC_UB
AUC_LB_test = AUC_LB


# for test [2]: 
risk = f_get_risk_predictions(sess, model, tea_data, tea_data_mi, [pred_time], [eval_time])
risk = risk[0][:, 0, 0]

# we need: label, time
label = tea_label[:, 0]
time = tea_time[:, 0]
# true label: 
label_te = label * (time <= pred_time + eval_time)

# we need a discretised scale from 0 to 1
min_risk = min(risk)
max_risk = max(risk)
step = (max_risk - min_risk)/steps #step width
r = [min_risk + step * i for i in range(steps)]
r = r[1:len(r)]


# at each scale, calculate sens and spec
Lsens = []
Lspec = []
LPPV = []
LNPV = []
for ri in r: 
    label_pred = risk >= ri # predicted label
    sens = sum(label_pred * label_te)/sum(label_te)
    spec = 1 - sum((1 - label_pred) * (1 - label_te))/sum(1 - label_te)
    PPV = sum(label_pred * label_te)/sum(label_pred)
    NPV = sum((1 - label_pred) * (1 - label_te))/sum(1 - label_pred)
    Lsens.append(sens)
    Lspec.append(spec)
    LPPV.append(PPV)
    LNPV.append(NPV)
# print(Lsens)
# print(Lspec)

# get AUC with trapezium rule
'''
rL = len(r) - 1
AUCL = []
for i in list(range(rL)): 
    AUCL.append(1/2 * (Lsens[i] + Lsens[i + 1]) * (Lspec[i + 1] - Lspec[i]))

AUC = - sum(AUCL)
'''
# here, an alternative using the stat_util.py

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import stat_util    #Compute AUC with 95% confidence interval

score, ci_lower, ci_upper, scores = stat_util.score_ci(label_te, risk, score_fun=roc_auc_score,seed=142857)


AUC_name = str(np.round(score, decimals = 3))
AUC_UB = str(np.round(ci_upper, decimals = 3))
AUC_LB = str(np.round(ci_lower, decimals = 3))
print("[Test set 2] Time-varying AUC at landmark " + str(eval_time) + " with horizon " + str(pred_time) + ": " + AUC_name + " (" + AUC_LB + ", " + AUC_UB + ")")

# store results
# firstly, deal with the fucking disgusting float32 stuff
Lspec_to_save = [float(i) for i in Lspec]
Lsens_to_save = [float(i) for i in Lsens]
AUC = score
AUC_to_save = float(AUC)
r_to_save = [float(i) for i in r]
tv_te_log = {"spec": Lspec_to_save, 
"sens": Lsens_to_save, 
"AUC": AUC_to_save, 
"steps": r_to_save}

# eval_path = target_dir + '/eval'
tvROC_dir = target_dir + '/eval/tvROC/'

if not os.path.exists(tvROC_dir):
    os.makedirs(tvROC_dir)

landmark_horizon_lab = 'L' + str(eval_time) + 'H' + str(pred_time)
with open(tvROC_dir + landmark_horizon_lab + '_log_test_2.json', "w") as f:
    json.dump(tv_te_log, f)

# plot bit
Lspec_testa = Lspec
Lsens_testa = Lsens
AUC_name_testa = AUC_name
AUC_UB_testa = AUC_UB
AUC_LB_testa = AUC_LB

Fig_name = tvROC_dir + landmark_horizon_lab + '_tvROC_general.png'
from matplotlib import pyplot as plt


f = plt.figure()
f.set_figwidth(6)
f.set_figheight(6)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.xlim((0, 1.05))
plt.ylim((0, 1.05)) # force plt to display the full plot! 
plt.title('Landmark Time: ' + str(eval_time) + '; Horizon Time: '+ str(pred_time))
plt.text(x = 0.25, y = 0.2, s = train_name + "_tvAUC: "+ AUC_name_train + " (" + AUC_LB_train + ", " + AUC_UB_train + ")")
plt.text(x = 0.25, y = 0.16, s = test1_name + "_tvAUC: "+ AUC_name_test + " (" + AUC_LB_test + ", " + AUC_UB_test + ")")
plt.text(x = 0.25, y = 0.12, s = test2_name + "_tvAUC: "+ AUC_name_testa + " (" + AUC_LB_testa + ", " + AUC_UB_testa + ")")

plt.plot(Lspec_train, Lsens_train, label = train_name)
plt.plot(Lspec_test, Lsens_test, label = test1_name)
plt.plot(Lspec_testa, Lsens_testa, label = test2_name)

plt.plot([0, 1], [0, 1],color='navy', lw=1, linestyle='--')

plt.legend([train_name, test1_name, test2_name], loc = 'upper left')
# plt.show()
plt.savefig(Fig_name)