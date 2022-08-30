import pandas as pd
import numpy as np

import time

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,data_per_yearloan):
    # All variables that are imputed are treated as continious variables
    
    # var_mis_con: CONTINIOUS variables with missing values to be imputed
    # var_mis_cat: CATEGORICAL variables with missing values to be imputed
    # var_use: variables used in the vector space for the kNN algorithm for imputation 
    
    # Making sure that index is reset.
    # If not, the process will fail
    data_per_yearloan = data_per_yearloan.reset_index(drop=True) # Reset index

    # Checking that variables used for imputation do not have missing values
    temp = np.sum(pd.isnull(data_per_yearloan[var_use]))
    temp = temp[temp!=0].index.tolist()
    if len(temp)>=1:
        for i in temp:
            print('-------------------------')
            print('ERROR: Variable \'{}\' is used for k-NN imputation. However, it has missing values. Missing values are set to zero in the k-NN imputation, so consider not including this variable.'.format(i))
        print('-------------------------')
        print('Imputing failed. No values imputed.')
        print('-------------------------')

    elif len(temp)==0:
        # Imputing variable, one at a time

        # Creating vector space used for the kNN-algorithm
        data_var_use = data_per_yearloan[var_use]

        # Standardizing the vector space to a mean of 0 and 
        # standard deviation of 1 for each variable, respectively
        data_var_use = pd.DataFrame(StandardScaler().fit_transform(data_var_use),columns=data_var_use.columns)

        # First, continious variables
        for i in var_mis_con:
            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data_per_yearloan[i],data_var_use],axis=1)
            num_missing_before = np.sum(pd.isnull(data_per_yearloan[i]))

            # Imputing
            imputer = KNNImputer(n_neighbors=3,weights='distance')
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data_per_yearloan[i] = temp[0]

            num_missing_after = np.sum(pd.isnull(data_per_yearloan[i]))
            # print('Imputed {} instances of missing values for continious variable \'{}\''.format(num_missing_before-num_missing_after,i))
            # print('Elapset time: {} minutes'.format(np.round((time.time()-t)/60,2)))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

        # Second, categorical variables including dummies
        for i in var_mis_cat:
            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data_per_yearloan[i],data_var_use],axis=1)
            num_missing_before = np.sum(pd.isnull(data_per_yearloan[i]))
            num_cat_values_before = len(data_per_yearloan[i][pd.isnull(data_per_yearloan[i])==False].unique())
            
            # Imputing
            imputer = KNNImputer(n_neighbors=1)
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data_per_yearloan[i] = temp[0]

            num_missing_after = np.sum(pd.isnull(data_per_yearloan[i]))
            num_cat_values_after = len(data_per_yearloan[i][pd.isnull(data_per_yearloan[i])==False].unique())

            if num_cat_values_before!=num_cat_values_after:
                print('ERROR: number of categorical values before and after impotation are {} and {}, respectively.'.format(num_cat_values_before,num_cat_values_after))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

    return data_per_yearloan

def sample_rrw(data,response_variable):

    unique_years = data['yearloan'].unique()

    # Shuffle entire DataFrame and reset index
    data = data.sample(frac=1,random_state=0).reset_index(drop=True)

    list_true   = data[data[response_variable]==1].index.tolist()

    data_false = data[data[response_variable]==0]

    list_false = []
    for i in range(len(unique_years)):
        temp = data_false[data_false['yearloan']==unique_years[i]].index.tolist()
        list_false = list_false+temp[0:int(np.round(len(temp)*0.01))]
    
    if len(np.unique(list_false))!=len(list_false):
        print('ERROR in sample(): not all in list_false are unique')

    data_to_use = data.loc[list_true+list_false]
    data_to_use = data_to_use.reset_index(drop=True) # Reset index

    return data_to_use

def sample_rrw_balancing(data,random_seed_for_sampling,response_variable):
    # Removing 15 variables with npl==0 and missing city dummies
    ind = (pd.isnull(data.istanbuldummy))|(pd.isnull(data.izmirdummy))|(pd.isnull(data.ankaradummy))
    data = data[ind==False]
    data = data.reset_index(drop=True) # Reset index

    unique_years = np.sort(data['yearloan'].unique())

    # Shuffle entire DataFrame and reset index
    data = data.sample(frac=1,random_state=random_seed_for_sampling).reset_index(drop=True)

    for yearloan in unique_years:
        data_temp = (data[data['yearloan']==yearloan]).reset_index(drop=True)

        data_temp_npl = data_temp[data_temp[response_variable]==1]
        data_temp_non_npl = data_temp[data_temp[response_variable]==0]

        principal_npl     = data_temp_npl['principal']
        principal_non_npl = data_temp_non_npl['principal']

        for value in principal_npl:
            # Selects the one with lowest difference in terms of principal
            ind = principal_non_npl[np.abs(principal_non_npl-value)==np.min(np.abs(principal_non_npl-value))].index.values[0]

            data_temp_npl = data_temp_npl.append(data_temp_non_npl.loc[ind],ignore_index=True)
            data_temp_non_npl = data_temp_non_npl.drop(ind)
            principal_non_npl = principal_non_npl.drop(ind)

        # Adding all data together
        if yearloan == unique_years[0]:
            data_sampled = data_temp_npl.copy()
        else:
            data_sampled = pd.concat([data_sampled,data_temp_npl])

    # Reset index 
    data_sampled = data_sampled.reset_index(drop=True)

    return data_sampled


def handling_missing_values(data):
    if False:
        # Show missing values
        temp_data = data[data.yearloan<=2009]
        temp = np.sum(pd.isnull(temp_data))
        temp[temp!=0]/temp_data.shape[0]

        # Per year:
        pd.pivot_table(\
            pd.concat([pd.isnull(data),pd.Series(data['yearloan'],name='aar')],axis=1),\
            values='genderd',\
            index=['aar'],\
            aggfunc=np.sum)
        # Fraction per year
        pd.pivot_table(\
            pd.concat([pd.isnull(data),pd.Series(data['yearloan'],name='aar')],axis=1),\
            values='genderd',\
            index=['aar'],\
            # columns=['C'],\
            aggfunc=np.mean)

    # CONTINIOUS variables with missing values to be imputed
    var_mis_con = [
        'age',
    ]

    # CATEGORICAL variables with missing values to be imputed
    var_mis_cat = [
        'genderd',
        'marriedd',
        'dbank',
        'daccounting',
        'dbroker',
        'dinsurance',
        'dbusecon',
        'dotherfin',
        'dgm',
        'dboard',
        'downer',
        'elementd',
        'highsd',
        'colleged',
        'masterphdd',
    ]

    # variables used in the vector space for the kNN algorithm for imputation 
    var_use = [
        'principal',
        'installment',
    ]

    num_obs_before = data.shape[0]
    check_sum_before = np.sum(data['customerno'])

    data_without_missing = pd.DataFrame(columns=data.columns)
    for yearloan in np.sort(data['yearloan'].unique()):
        data_per_yearloan = data[data['yearloan']==yearloan]
        data_per_yearloan = KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,data_per_yearloan)
        data_without_missing = pd.concat([data_without_missing,data_per_yearloan])

    if num_obs_before != data_without_missing.shape[0]:
        print('ERROR: handling missing values failed')
    if check_sum_before != np.sum(data_without_missing['customerno']):
        print('ERROR: handling missing values failed')

    if np.sum(np.sum(pd.isnull(data_without_missing)))!=0:
        print('ERROR: There are still missing values in the data.')

    return data_without_missing

def printing_number_of_observations_per_year(data,response_variable):
    for yearloan in np.sort(data.yearloan.unique()):
        temp = data[data['yearloan']==yearloan]
        num_obs = temp.shape[0]
        num_def = np.sum(temp[response_variable])
        def_freq = np.round(100*num_def/num_obs,2)
        print('{} - num_obs: {} - num_def: {} - def_freq: {}'.format(yearloan,num_obs,num_def,def_freq))
    num_obs = data.shape[0]
    num_def = np.sum(data[response_variable])
    def_freq = np.round(100*num_def/num_obs,2)
    print('Total - num_obs: {} - num_def: {} - def_freq: {}'.format(num_obs,num_def,def_freq))

    for yearloan in np.sort(data_sampled.yearloan.unique()):
        temp = data_sampled[data_sampled['yearloan']==yearloan]
        num_obs = temp.shape[0]
        num_def = np.sum(temp[response_variable])
        def_freq = np.round(100*num_def/num_obs,2)
        print('{} - num_obs: {} - num_def: {} - def_freq: {}'.format(yearloan,num_obs,num_def,def_freq))
    num_obs = data_sampled.shape[0]
    num_def = np.sum(data[response_variable])
    def_freq = np.round(100*num_def/num_obs,2)
    print('Total - num_obs: {} - num_def: {} - def_freq: {}'.format(num_obs,num_def,def_freq))
