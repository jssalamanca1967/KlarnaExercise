# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:59:39 2019

@author: johns
"""

import numpy as np
import pandas as pd
import pandas_profiling 
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv",sep=";")

dataset_model = dataset[dataset["default"].isnull() == False]

# Univariate Analysis
# report = pandas_profiling.ProfileReport(dataset_model)
# report.to_file("report.html")

def var_num_to_cat(data, var_name, inter, labs):
    
    cutdone = pd.cut(data[var_name], bins = inter)
    
    newvar = pd.DataFrame(np.array(cutdone), columns = [var_name + "_cat"])
    
    inter[0]
    
    for i in range(0,len(inter)):
        newvar[newvar[var_name + "_cat"] == inter[i]] = labs[i]
    
    newvar[var_name + "_cat"] = newvar[var_name + "_cat"].astype('category')
    
    return newvar

dataset_model_age_cat = pd.get_dummies(dataset_model, columns = ["age_cat"], prefix = ["age_cat"])
dataset_model.age_cat.cat.codes

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

def cat_sum_default(data, cat_name, response_name): 
    data[cat_name] = data[cat_name].fillna("NoData")
    grouped = data.groupby(cat_name)
    grouped_sum_def = grouped[response_name].agg(np.sum)
    grouped_count = grouped.size()
    freq = grouped_count.values/sum(grouped_count.values) * 100
    
    np_grouped = np.array([grouped_sum_def.index, grouped_sum_def.values, freq])
    np_grouped = np_grouped.transpose()
    
    df_grouped = pd.DataFrame(np_grouped, columns = ["Categories", "Default", "Frequency"])

    print(df_grouped)
    
    fig = plt.figure() # Create matplotlib figure
    
    objects = df_grouped.Categories
    y_pos = np.arange(len(objects))

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
    
    width = 0.4
    
    df_grouped.Default.plot(kind='bar', color='red', ax=ax, width=width, position=1)
    df_grouped.Frequency.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
    
    plt.xticks(y_pos, objects, rotation=0)
    ax.set_ylabel('Default Sum')
    ax2.set_ylabel('Frequency')
    
    plt.show()
    # %matplotlib auto
    plt.savefig('Graph_SumDefault/SumDefault_' + grouped_sum_def.index.name + '.png')


cat_sum_default(dataset_model, 'account_status', 'default')
cat_sum_default(dataset_model, 'account_worst_status_0_3m', 'default')
cat_sum_default(dataset_model, 'account_worst_status_12_24m', 'default')
cat_sum_default(dataset_model, 'account_worst_status_3_6m', 'default')
cat_sum_default(dataset_model, 'account_worst_status_6_12m', 'default')
# cat_sum_default(dataset_model, 'merchant_category', 'default')
cat_sum_default(dataset_model, 'merchant_group', 'default')
cat_sum_default(dataset_model, 'has_paid', 'default')
cat_sum_default(dataset_model, 'name_in_email', 'default')
cat_sum_default(dataset_model, 'status_last_archived_0_24m', 'default')
cat_sum_default(dataset_model, 'status_2nd_last_archived_0_24m', 'default')
cat_sum_default(dataset_model, 'status_3rd_last_archived_0_24m', 'default')
cat_sum_default(dataset_model, 'status_max_archived_0_6_months', 'default')
cat_sum_default(dataset_model, 'status_max_archived_0_12_months', 'default')
cat_sum_default(dataset_model, 'status_max_archived_0_24_months', 'default')
cat_sum_default(dataset_model, 'worst_status_active_inv', 'default')

cat_sum_default(dataset_model, 'account_amount_added_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'account_days_in_dc_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'account_days_in_rem_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'account_days_in_term_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'account_incoming_debt_vs_paid_0_24m_cat', 'default')
cat_sum_default(dataset_model, 'age_cat', 'default')
cat_sum_default(dataset_model, 'num_active_inv_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_dc_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_dc_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_ok_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_ok_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_rem_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_written_off_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'num_arch_written_off_12_24m_cat', 'default')
cat_sum_default(dataset_model, 'num_unpaid_bills_cat', 'default')
cat_sum_default(dataset_model, 'recovery_debt_cat', 'default')
cat_sum_default(dataset_model, 'sum_capital_paid_account_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'sum_capital_paid_account_12_24m_cat', 'default')

# Numeric Variables turned into categorical

cat_sum_default(dataset_model, 'avg_payment_span_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'avg_payment_span_0_3m_cat', 'default')
cat_sum_default(dataset_model, 'max_paid_inv_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'max_paid_inv_0_24m_cat', 'default')
cat_sum_default(dataset_model, 'num_active_div_by_paid_inv_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'sum_paid_inv_0_12m_cat', 'default')
cat_sum_default(dataset_model, 'time_hours_cat', 'default')


