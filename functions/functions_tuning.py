import pandas as pd
import numpy as np

import xgboost

import os
import time
import pickle

from sklearn.svm import l1_min_c
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
# © 2007 - 2019, scikit-learn developers (BSD License).

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

import lightgbm as lgb


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def tuning_rrw(method_version,X_train,y_train):
    
    y_train = y_train.astype(int)
    X_train = X_train.astype(float)

    ##################################################################
    if method_version=='LR':
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)
        
        num = 20
        start = 0
        stop = 7
        cs = l1_min_c(X_train, y_train, loss='log')*np.logspace(start, stop, num)
        
        param_grid = [{\
            'C': cs,\
            'solver': ['saga','liblinear'],\
            }]
        model = linear_model.LogisticRegression(\
            penalty='l1',
            max_iter=1e6,
            fit_intercept=True,
            random_state=0)
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version == 'XGBoost':
        param_grid = {
            'learning_rate': (0.1,0.2,0.3),
            'n_estimators': (50,100,150),
            'max_depth': (3,5,7),
            'subsample':  (0.5,0.75,1), 
            'colsample_bytree': (0.5,0.75,1),
            'min_child_weight':(1,3,5)
        }
        model = xgboost.XGBClassifier(
                gamma=0,
                objective= 'binary:logistic',
                random_state=1,
                use_label_encoder=False
                )
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train,y_train,eval_metric='auc')
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='DT':
        param_grid = [{\
            'criterion': ('gini','entropy'),\
            'splitter': ('best','random'),\
            'max_depth': (5,7,9,None),\
            'min_samples_split': (1,2,3),\
            'min_samples_leaf': (1,2,3),\
            }]
        model = tree.DecisionTreeClassifier(\
            random_state=0,\
            max_leaf_nodes=None,\
            max_features=None)
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='RF':
        param_grid = [{\
            'criterion': ('gini','entropy'),\
            'n_estimators': (50,100,150),\
            'max_depth': (5,7,9,None),\
            }]
        model = RandomForestClassifier(\
            random_state=0,\
            warm_start=False,\
            bootstrap=True,\
            max_features=None)
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='CatBoost':
        param_grid = [{\
            'iterations': (50,100,150),\
            'learning_rate': (0.1,0.2,0.3),\
            'depth': (5,7,9),\
            'l2_leaf_reg': (1, 3, 5, 9),\
            }]
        model = CatBoostClassifier(\
            verbose=False,\
            random_seed=0)
        temp = model.grid_search(param_grid,X=X_train,y=y_train,\
            plot=False,\
            cv=3,\
            partition_random_seed=0,\
            verbose=False,\
            calc_cv_statistics=False)
        best_parameters = temp['params']

    ##################################################################
    elif method_version == 'LightGBM':
        param_grid = {
            'learning_rate': (0.1,0.2,0.3),
            'n_estimators': (50,100,150),
            'max_depth': (-1,3,5,7),
            'subsample':  (0.5,0.75,1), 
            'colsample_bytree': (0.5,0.75,1),
            'min_child_weight':(1e-3,1e-2,1e-1,1,10)
        }
        model = lgb.LGBMClassifier(random_state=0)
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train,y_train,eval_metric='auc')
        best_parameters = grid_search.best_params_

    ##################################################################
    else:
        print('ERROR: Wrongly defined method_version')

    return best_parameters


def tune_models(data,method_versions_to_tune,folder_name,test_years,response_variable,explanatory_variables_all):
    # Making folder for saving best_hyper_parameters_dict_all
    make_folder(folder_name)

    t_total = time.time()
    for method_version in method_versions_to_tune:
        best_parameters_dict_method = {}

        for year in test_years:
                t_year = time.time()
        
                data_train  = data[data['yearloan']<year]

                y_train = data_train[response_variable]

                X_train = data_train[explanatory_variables_all]
                
                best_parameters = tuning_rrw(method_version,X_train,y_train)
                best_parameters_dict_method[year] = best_parameters
                print('Elapset time tuning method {} for year {}: {} minutes'.format(method_version,year,np.round(((time.time() - t_year))/60,2)))
        
        # Saving best parameters for method
        f = open(folder_name+'/'+method_version+'.pkl','wb')
        pickle.dump(best_parameters_dict_method,f)
        f.close()

    print('Elapset time tuning total: {} minutes'.format(np.round(((time.time() - t_total))/60,2)))

def load_best_hyper_parameters(folder_name,method_versions_to_tune):
    # Load previously made dict with all best hyper-parameters
    best_hyper_parameters_dict_all = {}

    for method_version in method_versions_to_tune:
        f = open(folder_name+'/'+method_version+'.pkl', 'rb')
        best_hyper_parameters_dict_all[method_version] = pickle.load(f)
        f.close()

    return best_hyper_parameters_dict_all


def save_hyperparameters_selected_after_tuning_in_excel(best_hyper_parameters_all,best_hyper_parameters_sampled,test_years):
    folder_name = '../results'
    make_folder(folder_name)

    tree_methods =[
        'DT',
        'RF',
        'CatBoost',
        'XGBoost',
        'LightGBM',
    ]

    # All
    best_parameters_df_total = pd.DataFrame()
    for year in test_years:
        best_parameters_df = pd.DataFrame()
        for method_version in tree_methods:
            # Add to data frame of best parameters
            best_parameters = pd.Series(best_hyper_parameters_all[method_version][year],dtype='object')
            for i in best_parameters.index:
                if (i in best_parameters_df.index)==False:
                    #best_parameters_df = best_parameters_df.append(pd.Series(name=i,dtype='object'))
                    best_parameters_df = pd.concat([best_parameters_df, pd.DataFrame(index=[i])],axis=0)
            best_parameters_df[method_version] = best_parameters
        best_parameters_df.to_excel(folder_name+'/best-parameters_all_'+str(year)+'.xlsx',index=True)
        best_parameters_df_total = pd.concat([best_parameters_df_total,best_parameters_df],axis=0)
    best_parameters_df_total.to_excel(folder_name+'/best-parameters_all_total.xlsx',index=True)

    # Sampled
    best_parameters_df_total = pd.DataFrame()
    for year in test_years:
        best_parameters_df = pd.DataFrame()
        for method_version in tree_methods:
            # Add to data frame of best parameters
            best_parameters = pd.Series(best_hyper_parameters_sampled[method_version][year],dtype='object')
            for i in best_parameters.index:
                if (i in best_parameters_df.index)==False:
                    #best_parameters_df = best_parameters_df.append(pd.Series(name=i,dtype='object'))
                    best_parameters_df = pd.concat([best_parameters_df, pd.DataFrame(index=[i])],axis=0)
            best_parameters_df[method_version] = best_parameters
        best_parameters_df.to_excel(folder_name+'/best-parameters_sampled_'+str(year)+'.xlsx',index=True)
        best_parameters_df_total = pd.concat([best_parameters_df_total,best_parameters_df],axis=0)
    best_parameters_df_total.to_excel(folder_name+'/best-parameters_sampled_total.xlsx',index=True)
