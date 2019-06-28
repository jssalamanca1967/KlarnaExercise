# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:14:20 2019

@author: johns
"""

import numpy as np
import pandas as pd
import pandas_profiling 
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import sklearn as sk
import keras_metrics
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 

dataset = pd.read_csv("dataset.csv",sep=";")

dataset_model = dataset[dataset["default"].isnull() == False]


def var_num_to_cat(data, var_name, inter, labs):
    
    cutdone = pd.cut(data[var_name], bins = inter)
    
    newvar = pd.DataFrame(np.array(cutdone), columns = [var_name + "_cat"])
    
    inter[0]
    
    for i in range(0,len(inter)):
        newvar[newvar[var_name + "_cat"] == inter[i]] = labs[i]
    
    newvar[var_name + "_cat"] = newvar[var_name + "_cat"].fillna('NoData')
    newvar[var_name + "_cat"] = newvar[var_name + "_cat"].astype('category')
    
    return newvar

# Creation of categoric variables

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.account_amount_added_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'account_amount_added_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.account_days_in_dc_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'account_days_in_dc_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.account_days_in_rem_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'account_days_in_rem_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.account_days_in_term_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'account_days_in_term_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.account_incoming_debt_vs_paid_0_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'account_incoming_debt_vs_paid_0_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(17,30), (30,40), (40,50), (50,60),(60,max(dataset_model.age))])
labs = ['18-30','30-40','40-50','50-60','60+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'age', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_active_inv))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_active_inv', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_dc_0_12m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_dc_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_dc_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_dc_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_ok_0_12m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_ok_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_ok_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_ok_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_rem_0_12m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_rem_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_written_off_0_12m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_written_off_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_arch_written_off_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_arch_written_off_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.num_unpaid_bills))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_unpaid_bills', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.recovery_debt))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'recovery_debt', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.sum_capital_paid_account_0_12m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'sum_capital_paid_account_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_model.sum_capital_paid_account_12_24m))])
labs = ['0', '> 0']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'sum_capital_paid_account_12_24m', inter, labs)], axis = 1)

# Categoric from numeric variables

np.quantile(dataset_model.avg_payment_span_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,7),(7, 10), (10, 12), (12, 15), (15, 20), (20,32), (32,max(dataset_model.avg_payment_span_0_12m))])
labs = ['0-7', '7-10', '10-12', '12-15', '15-20', '20-32', '32+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'avg_payment_span_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_model.avg_payment_span_0_3m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,5),(5, 7), (7, 10), (10, 13), (13, 15), (15, 20), (20,max(dataset_model.avg_payment_span_0_3m))])
labs = ['0-5', '5-7', '7-10', '10-13', '13-15', '15-20', '20+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'avg_payment_span_0_3m', inter, labs)], axis = 1)

np.quantile(dataset_model.max_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 5000), (5000, 7000), (7000, 12000), (12000, 20000), (20000,max(dataset_model.max_paid_inv_0_12m))])
labs = ['0', '0-5k', '5k-7k', '7k-12k', '12k-20k', '20k+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'max_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_model.max_paid_inv_0_24m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 5000), (5000, 7000), (7000, 12000), (12000, 20000), (20000,max(dataset_model.max_paid_inv_0_24m))])
labs = ['0', '0-5k', '5k-7k', '7k-12k', '12k-20k', '20k+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'max_paid_inv_0_24m', inter, labs)], axis = 1)

np.quantile(dataset_model.num_active_div_by_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 0.2), (0.2, max(dataset_model.num_active_div_by_paid_inv_0_12m))])
labs = ['0', '0-0.2', '0.2+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'num_active_div_by_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_model.sum_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 7000), (7000, 15000), (15000, 30000), (30000, 90000), (90000, max(dataset_model.sum_paid_inv_0_12m))])
labs = ['0', '0-7k', '7k-15k', '15k-30k', '30k-90k', '90k+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'sum_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_model.time_hours.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,10), (10, 15), (15, 20), (20, max(dataset_model.time_hours))])
labs = ['0-10', '10-15', '15-20', '20+']
dataset_model = pd.concat([dataset_model, var_num_to_cat(dataset_model, 'time_hours', inter, labs)], axis = 1)

def dummies_creation(data):
    data = pd.get_dummies(data, columns = ['account_amount_added_12_24m_cat'], prefix = ['account_amount_added_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['account_days_in_dc_12_24m_cat'], prefix = ['account_days_in_dc_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['account_days_in_rem_12_24m_cat'], prefix = ['account_days_in_rem_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['account_days_in_term_12_24m_cat'], prefix = ['account_days_in_term_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['account_status'], prefix = ['account_status'])
    data = pd.get_dummies(data, columns = ['age_cat'], prefix = ['age_cat'])
    data = pd.get_dummies(data, columns = ['merchant_group'], prefix = ['merchant_group'])
    data = pd.get_dummies(data, columns = ['has_paid'], prefix = ['has_paid'])
    data = pd.get_dummies(data, columns = ['max_paid_inv_0_12m_cat'], prefix = ['max_paid_inv_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['max_paid_inv_0_24m_cat'], prefix = ['max_paid_inv_0_24m_cat'])
    data = pd.get_dummies(data, columns = ['name_in_email'], prefix = ['name_in_email'])
    data = pd.get_dummies(data, columns = ['num_active_div_by_paid_inv_0_12m_cat'], prefix = ['num_active_div_by_paid_inv_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['num_active_inv_cat'], prefix = ['num_active_inv_cat'])
    data = pd.get_dummies(data, columns = ['num_arch_dc_0_12m_cat'], prefix = ['num_arch_dc_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['num_arch_dc_12_24m_cat'], prefix = ['num_arch_dc_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['num_arch_ok_0_12m_cat'], prefix = ['num_arch_ok_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['num_arch_ok_12_24m_cat'], prefix = ['num_arch_ok_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['num_arch_rem_0_12m_cat'], prefix = ['num_arch_rem_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['num_unpaid_bills_cat'], prefix = ['num_unpaid_bills_cat'])
    data = pd.get_dummies(data, columns = ['status_last_archived_0_24m'], prefix = ['status_last_archived_0_24m'])
    data = pd.get_dummies(data, columns = ['status_2nd_last_archived_0_24m'], prefix = ['status_2nd_last_archived_0_24m'])
    data = pd.get_dummies(data, columns = ['status_3rd_last_archived_0_24m'], prefix = ['status_3rd_last_archived_0_24m'])
    data = pd.get_dummies(data, columns = ['status_max_archived_0_6_months'], prefix = ['status_max_archived_0_6_months'])
    data = pd.get_dummies(data, columns = ['status_max_archived_0_12_months'], prefix = ['status_max_archived_0_12_months'])
    data = pd.get_dummies(data, columns = ['status_max_archived_0_24_months'], prefix = ['status_max_archived_0_24_months'])
    data = pd.get_dummies(data, columns = ['sum_capital_paid_account_0_12m_cat'], prefix = ['sum_capital_paid_account_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['sum_capital_paid_account_12_24m_cat'], prefix = ['sum_capital_paid_account_12_24m_cat'])
    data = pd.get_dummies(data, columns = ['sum_paid_inv_0_12m_cat'], prefix = ['sum_paid_inv_0_12m_cat'])
    data = pd.get_dummies(data, columns = ['time_hours_cat'], prefix = ['time_hours_cat'])

    return data

dataset_testing = dummies_creation(dataset_model)

# 
dataset_testing =  pd.concat([dataset_testing.iloc[:,0:2], dataset_testing.iloc[:,39:150]], axis = 1)

# Labels are the values we want to predict
labels = np.array(dataset_testing['default'])
# Remove the labels from the features
# axis 1 refers to the columns
features= dataset_testing.drop(['default', 'uuid'], axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

train_dat = pd.DataFrame(np.append(train_features, train_labels[:,None], axis = 1), columns = np.append(feature_list, 'default'))

train_dat_1s = train_dat[train_dat['default'] == 1]
train_dat_0s = train_dat[train_dat['default'] == 0]
keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])

train_dat = pd.concat([keep_0s,train_dat_1s],axis=0)

train_features = np.array(train_dat.drop(['default'], axis = 1))
train_labels = np.array(train_dat['default'])

## NN Model

dummy_y_train= np_utils.to_categorical(train_labels)
dummy_y_test= np_utils.to_categorical(test_labels)

model=Sequential()
model.add(Dense(111, input_dim=111, activation="relu"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(500, activation="relu"))
model.add(Dense(100, activation="sigmoid"))
model.add(Dense(2,activation="softmax"))

def new_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
    
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy", keras_metrics.categorical_recall()])
    
model.fit(train_features, dummy_y_train, epochs=10, verbose=1)

sum(dummy_y_train[:,1])

predictions_train = model.predict_classes(train_features)
predictions_test = model.predict_classes(test_features)

probs_train = model.predict(train_features)
probs_test = model.predict(test_features)

conf_m_train_rf = sk.metrics.confusion_matrix(train_labels, predictions_train, labels=None, sample_weight=None)
print(conf_m_train_rf)

conf_m_test_rf = sk.metrics.confusion_matrix(test_labels, predictions_test, labels=None, sample_weight=None)
print(conf_m_test_rf)

print("Train Accuracy:", sk.metrics.accuracy_score(train_labels, predictions_train))
print("Test Accuracy:", sk.metrics.accuracy_score(test_labels, predictions_test))

print("Train Recall:", sk.metrics.recall_score(train_labels, predictions_train))
print("Test Recall:", sk.metrics.recall_score(test_labels, predictions_test))

model.save(filepath = "model.pkl")

