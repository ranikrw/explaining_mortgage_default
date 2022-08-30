import pandas as pd
import numpy as np
import time

import os

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'functions')
from functions_methods import *
from functions_process_data import *
from functions_tuning import *

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

##################################################################
##  Load data                                                   ##
##################################################################
data = pd.read_csv('../data.csv',sep=';',index_col=0)
data = data.reset_index(drop=True) # Reset index

##################################################################
##  Defining explanatory variables                              ##
##################################################################
response_variable = 'npl'

explanatory_variables_all = []

# Loan-specific variables
explanatory_variables_all = explanatory_variables_all + [\
    'principal',\
    'pctloan',\
    'installment',\
    'pctinstallment',\
    'monthloan',\
    'ageloan',\
    ]

# Demographic variables
explanatory_variables_all = explanatory_variables_all + [\
    'genderd',\
    'marriedd',\
    'dbank',\
    'daccounting',\
    'dbroker',\
    'dinsurance',\
    'dbusecon',\
    'dotherfin',\
    'dgm',\
    'dboard',\
    'downer',\
    'elementd',\
    'highsd',\
    'colleged',\
    'masterphdd',\
    'istanbuldummy',\
    'ankaradummy',\
    'izmirdummy',\
    'age',\
    ]

# Macro variables
explanatory_variables_all = explanatory_variables_all + [\
    'cpianch',\
    'uneprc',\
    'holoanch',\
    'fxrate',\
    'intrate',\
    'gdp',\
    'gdppercap',\
    'consconfindex',\
    'constind',\
    'chgmar',\
    ]

print('Number of mortgages: {}'.format(data.shape[0]))
print('Number of mortgages categorized as default: {}'.format(np.sum(data[response_variable])))

##################################################################
##  Handling missing values                                     ##
##################################################################
print('Mortgages with missing values before imputing: {}%'.format(np.round(100*np.max(np.sum(pd.isnull(data))/data.shape[0]),2)))
data = handling_missing_values(data)
print('Done imputing missing values')

##################################################################
##  Sampling                                                    ##
##################################################################
data_sampled = sample_rrw(data,response_variable)

##################################################################
##  Defining test years                                         ##
##################################################################
test_years = [2008,2009,2010]

##################################################################
##  Make descriptive statistics                                 ##
##################################################################
# Making folder for saving descriptives
make_folder('../descriptives')

# All data
descriptives = pd.DataFrame(index=explanatory_variables_all)
for i in explanatory_variables_all:
    descriptives.at[i,'Mean'] = data[i].mean()
    descriptives.at[i,'Median'] = data[i].median()
    descriptives.at[i,'Std'] = data[i].std()
descriptives.to_excel('descriptives/descriptives_full_data_set.xlsx')

# Sampled data
descriptives = pd.DataFrame(index=explanatory_variables_all)
for i in explanatory_variables_all:
    descriptives.at[i,'Mean'] = data_sampled[i].mean()
    descriptives.at[i,'Median'] = data_sampled[i].median()
    descriptives.at[i,'Std'] = data_sampled[i].std()
descriptives.to_excel('descriptives/descriptives_data_sampled.xlsx')

# Print number of observations
printing_number_of_observations_per_year(data,response_variable)

##################################################################
## Tuning                                                       ##
##################################################################
method_versions_to_tune =[
    'DT',
    'RF',
    'CatBoost',
    'XGBoost',
    'LightGBM',
    'LR', # This is for finding the optimal lambda value for LASSO 
]

do_tuning = False 
# Set to True if tuning parameters
# Set to False to load tuned parameter values

# All data
folder_name = '../hyper_parameters_all_data'
if do_tuning:
    tune_models(data,method_versions_to_tune,folder_name,test_years,response_variable,explanatory_variables_all)
else:
    # Loading tuned hyperparameters:
    best_hyper_parameters_all = load_best_hyper_parameters(folder_name,method_versions_to_tune)

# Sampled data
folder_name = '../hyper_parameters_sampled_data'
if do_tuning:
    tune_models(data_sampled,method_versions_to_tune,folder_name,test_years,response_variable,explanatory_variables_all)
else:
    # Loading tuned hyperparameters:
    best_hyper_parameters_sampled = load_best_hyper_parameters(folder_name,method_versions_to_tune)

if do_tuning:
    # Saving hyperparameters selected after tuning in Excel
    save_hyperparameters_selected_after_tuning_in_excel(best_hyper_parameters_all,best_hyper_parameters_sampled,test_years)

##################################################################
##  Analysis                                                    ##
##################################################################
# Get evaluation metrics
evaluation_metrics =[\
    'AUC',\
    'Accuracy ratio',\
    'Brier score',\
    ]

method_versions =[
    'LR',
    'DT',
    'RF',
    'CatBoost',
    'XGBoost',
    'LightGBM',
]

# Making empty data frames for inserting results
TOTAL_results_in_sample_all         = pd.DataFrame()
TOTAL_results_out_of_sample_all     = pd.DataFrame()
TOTAL_results_in_sample_sampled     = pd.DataFrame()
TOTAL_results_out_of_sample_sampled = pd.DataFrame()

columns = []
for j in test_years:
    for i in method_versions[1:]:
        columns.append(i+'-'+str(j))
variable_importance_table_all       = pd.DataFrame([[None]*len(columns)]*len(explanatory_variables_all),index=explanatory_variables_all,columns=columns)
variable_importance_table_sampled   = pd.DataFrame([[None]*len(columns)]*len(explanatory_variables_all),index=explanatory_variables_all,columns=columns)

t_total = time.time()
for year in test_years:

    results_in_sample_all           = pd.DataFrame(index=[year]+evaluation_metrics)
    results_in_sample_sampled       = pd.DataFrame(index=[year]+evaluation_metrics)
    results_out_of_sample_all       = pd.DataFrame(index=[year]+evaluation_metrics)
    results_out_of_sample_sampled   = pd.DataFrame(index=[year]+evaluation_metrics)
    
    data_test           = data[data['yearloan']==year]
    data_train          = data[data['yearloan']<year]
    data_train_sampled  = data_sampled[data_sampled['yearloan']<year]

    for method_version in method_versions:

        series_in_sample_all,series_out_of_sample_all,series_in_sample_sampled,series_out_of_sample_sampled,variable_importance_table_all,variable_importance_table_sampled = model_and_get_results(method_version,data_train,data_train_sampled,data_test,explanatory_variables_all,response_variable,year,best_hyper_parameters_all,best_hyper_parameters_sampled,variable_importance_table_all,variable_importance_table_sampled)

        # Inserting results into DataFrames
        results_in_sample_all[method_version] = series_in_sample_all
        results_out_of_sample_all[method_version] = series_out_of_sample_all
        results_in_sample_sampled[method_version] = series_in_sample_sampled
        results_out_of_sample_sampled[method_version] = series_out_of_sample_sampled

    TOTAL_results_in_sample_all         = pd.concat([TOTAL_results_in_sample_all,results_in_sample_all])
    TOTAL_results_out_of_sample_all     = pd.concat([TOTAL_results_out_of_sample_all,results_out_of_sample_all])
    TOTAL_results_in_sample_sampled     = pd.concat([TOTAL_results_in_sample_sampled,results_in_sample_sampled])
    TOTAL_results_out_of_sample_sampled = pd.concat([TOTAL_results_out_of_sample_sampled,results_out_of_sample_sampled])

print('Elapset time: {} minutes'.format(np.round(((time.time() - t_total))/60,2)))

# Saving results
folder_name = '../results'
make_folder(folder_name)
TOTAL_results_in_sample_all.to_excel(folder_name+'/in-sample-fit - all.xlsx',index=True)
TOTAL_results_out_of_sample_all.to_excel(folder_name+'/out-of-sample - all.xlsx',index=True)
TOTAL_results_in_sample_sampled.to_excel(folder_name+'/in-sample-fit - sampled.xlsx',index=True)
TOTAL_results_out_of_sample_sampled.to_excel(folder_name+'/out-of-sample - sampled.xlsx',index=True)

variable_importance_table_all.to_excel(folder_name+'/variable_importance_table_all.xlsx',index=True)
variable_importance_table_sampled.to_excel(folder_name+'/variable_importance_table_sampled.xlsx',index=True)
