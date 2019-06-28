# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:42:23 2019

@author: johns
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("dataset.csv",sep=";")

dataset_prediction = dataset[dataset["default"].isnull() == True]
dataset_prediction.reset_index(drop=True, inplace=True)

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

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.account_amount_added_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'account_amount_added_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.account_days_in_dc_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'account_days_in_dc_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.account_days_in_rem_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'account_days_in_rem_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.account_days_in_term_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'account_days_in_term_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.account_incoming_debt_vs_paid_0_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'account_incoming_debt_vs_paid_0_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(17,30), (30,40), (40,50), (50,60),(60,max(dataset_prediction.age))])
labs = ['18-30','30-40','40-50','50-60','60+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'age', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_active_inv))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_active_inv', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_dc_0_12m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_dc_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_dc_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_dc_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_ok_0_12m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_ok_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_ok_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_ok_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_rem_0_12m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_rem_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_written_off_0_12m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_written_off_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_arch_written_off_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_arch_written_off_12_24m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.num_unpaid_bills))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_unpaid_bills', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.recovery_debt))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'recovery_debt', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.sum_capital_paid_account_0_12m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'sum_capital_paid_account_0_12m', inter, labs)], axis = 1)

inter = pd.IntervalIndex.from_tuples([(-1,0),(1,max(dataset_prediction.sum_capital_paid_account_12_24m))])
labs = ['0', '> 0']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'sum_capital_paid_account_12_24m', inter, labs)], axis = 1)

# Categoric from numeric variables

np.quantile(dataset_prediction.avg_payment_span_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,7),(7, 10), (10, 12), (12, 15), (15, 20), (20,32), (32,max(dataset_prediction.avg_payment_span_0_12m))])
labs = ['0-7', '7-10', '10-12', '12-15', '15-20', '20-32', '32+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'avg_payment_span_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.avg_payment_span_0_3m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,5),(5, 7), (7, 10), (10, 13), (13, 15), (15, 20), (20,max(dataset_prediction.avg_payment_span_0_3m))])
labs = ['0-5', '5-7', '7-10', '10-13', '13-15', '15-20', '20+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'avg_payment_span_0_3m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.max_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 5000), (5000, 7000), (7000, 12000), (12000, 20000), (20000,max(dataset_prediction.max_paid_inv_0_12m))])
labs = ['0', '0-5k', '5k-7k', '7k-12k', '12k-20k', '20k+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'max_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.max_paid_inv_0_24m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 5000), (5000, 7000), (7000, 12000), (12000, 20000), (20000,max(dataset_prediction.max_paid_inv_0_24m))])
labs = ['0', '0-5k', '5k-7k', '7k-12k', '12k-20k', '20k+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'max_paid_inv_0_24m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.num_active_div_by_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 0.2), (0.2, max(dataset_prediction.num_active_div_by_paid_inv_0_12m))])
labs = ['0', '0-0.2', '0.2+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'num_active_div_by_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.sum_paid_inv_0_12m.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,0), (0, 7000), (7000, 15000), (15000, 30000), (30000, 90000), (90000, max(dataset_prediction.sum_paid_inv_0_12m))])
labs = ['0', '0-7k', '7k-15k', '15k-30k', '30k-90k', '90k+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'sum_paid_inv_0_12m', inter, labs)], axis = 1)

np.quantile(dataset_prediction.time_hours.dropna(how='any',axis=0) , q = np.linspace(0,1,num = 21))
inter = pd.IntervalIndex.from_tuples([(-1,10), (10, 15), (15, 20), (20, max(dataset_prediction.time_hours))])
labs = ['0-10', '10-15', '15-20', '20+']
dataset_prediction = pd.concat([dataset_prediction, var_num_to_cat(dataset_prediction, 'time_hours', inter, labs)], axis = 1)

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

dataset_final = dummies_creation(dataset_prediction)

# dataset_testing =  pd.concat([dataset_testing.iloc[:,0:2], dataset_testing.iloc[:,27:235]], axis = 1)
dataset_final =  pd.concat([dataset_final.iloc[:,0:2], dataset_final.iloc[:,39:150]], axis = 1)

X = dataset_final.drop(['default', 'uuid'], axis = 1)
uuid = dataset_final['uuid']
# Saving feature names for later use
X_Names = list(X.columns)
# Convert to numpy array
X = np.array(X)

probs_resp = model.predict(X)

probs_true = probs_resp[:,1]

probs_true = pd.DataFrame(probs_true, columns = ["pd"])
response = pd.concat([uuid, probs_true], axis = 1)

response.to_csv("Answer.csv", sep=';', index = None)
