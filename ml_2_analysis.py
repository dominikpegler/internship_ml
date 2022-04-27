# -*- coding: utf-8 -*-
"""
Machine learning based data analysis
v352
@author: Dr. David Steyrl david.steyrl@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shap
import warnings
from collections import Counter
from itertools import combinations
from joblib import delayed
from joblib import Parallel
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from skopt.plots import plot_convergence
from skopt.plots import plot_evaluations
from skopt.plots import plot_objective
from skopt.space import Integer
from skopt.space import Real
from time import time

warnings_string = (
    'ignore:Using categorical_feature in Dataset.:::, ' +
    'ignore:Objective did not converge.:::, ' +
    'ignore:The objective has been evaluated at this point before.:::')
os.environ["PYTHONWARNINGS"] = warnings_string

warnings.filterwarnings(
    'ignore',
    'The objective has been evaluated at this point before.')
warnings.filterwarnings(
    'ignore',
    'Using categorical_feature in Dataset.')
warnings.filterwarnings(
    'ignore',
    'A worker stopped while some jobs were given to the executor.')


def create_dir(path):
    """
    Create specified directory if not existing.

    Parameters
    ----------
    path : string
        Path to to check to be created.

    Returns
    -------
    None.

    """
    # Check if dir exists
    if not os.path.isdir(path):
        # Create dir
        os.mkdir(path)


def get_n_rep_cv(n_rep_cv, tst_frac, g_cv, coverage):
    """
    Get number of outer CV repetitions. 'auto' adjusts number of repetitions
    so that number of predictions approximately equals coverage times the
    number of available samples and guarantees a minimum of 5 repetitions.

    Parameters
    ----------
    n_rep_cv : str or int
        Str 'auto' or int that specifies the number of repetions directly.
    tst_frac : float
        Fraction of groups for testing.
    g_cv : datatframe
        Dataframe of groups.
    coverage : integer
        How often one sample is in test set on average.

    Returns
    -------
    n_rep_cv : int
        Specifies the number of cv repetions.

    """
    # If n_rep_cv is set to 'auto'
    if n_rep_cv == 'auto':
        # Get number of groups
        n_g_cv = g_cv.squeeze().unique().shape[0]
        # Calculate average group size
        avg_g_cv_size = g_cv.shape[0]/n_g_cv
        # Calculate average number of test samples per cv repetition
        n_tst_samples_per_cv_rep = int(tst_frac*g_cv.shape[0]*avg_g_cv_size)
        # Calculate number of cv reps to meet 3 times the number of samples
        n_rep_cv = int((g_cv.shape[0]*coverage)/n_tst_samples_per_cv_rep)
        # Guarantee min of 5 cv repetitions
        n_rep_cv = max(5, n_rep_cv)
        # Return n_rep_cv
        return n_rep_cv
    # If n_rep_cv is integer
    else:
        # Return n_rep_cv
        return n_rep_cv


def drop_nan_rows(g_cv, y, xn, xc):
    """
    Identify and drop rows containing nans from dataframes.

    Parameters
    ----------
    g_cv : dataframe
        Groups dataframe.
    y : dataframe
        Targets dataframe.
    xn : dataframe
        Numerical predictors dataframe.
    xc : dataframe
        Categorical predictors dataframe.

    Returns
    -------
    g_cv : dataframe
        Groups dataframe.
    y : dataframe
        Targets dataframe.
    xn : dataframe
        Numerical predictors dataframe.
    xc : dataframe
        Categorical predictors dataframe.

    """
    # Initialize list of rows with nans
    rows_nans = []
    # Search for nans in groups
    rows_nans.extend(list(g_cv.loc[g_cv.isna().any(1).to_numpy(), :].index))
    # Search for nans in targets
    rows_nans.extend(list(y.loc[y.isna().any(1).to_numpy(), :].index))
    # Search for nans in numeric predictors
    rows_nans.extend(list(xn.loc[xn.isna().any(1).to_numpy(), :].index))
    # Search for nans in categorical predictors
    rows_nans.extend(list(xc.loc[xc.isna().any(1).to_numpy(), :].index))
    # Drop rows from cv groups
    g_cv = g_cv.drop(rows_nans).reset_index(drop=True)
    # Drop rows from targets
    y = y.drop(rows_nans).reset_index(drop=True)
    # Check if numerical predictors is not empty
    if not xn.empty:
        # Drop rows from predictors
        xn = xn.drop(rows_nans).reset_index(drop=True)
    # Check if categorical predictors is not empty
    if not xc.empty:
        # Drop rows from predictors
        xc = xc.drop(rows_nans).reset_index(drop=True)

    # Drop rows and return results --------------------------------------------
    return g_cv, y, xn, xc


def ohe(xc):
    """
    One hot encode categorical data.

    Parameters
    ----------
    xc : dataframe
        Dataframe holding the categorical data.

    Returns
    -------
    xc_ohe : dataframe
        Dataframe holding the one hot encoded categorical data.

    """
    # Check if categorical predictors is not empty
    if not xc.empty:
        # Instanciate one hot encoder
        ohe = OneHotEncoder(categories='auto',
                            drop='if_binary',
                            sparse=False,
                            dtype=int,
                            handle_unknown='error')
        # One hot encode categorical predictors
        xc_ohe = pd.DataFrame(ohe.fit_transform(xc),
                              columns=ohe.get_feature_names_out())
    # If categorical predictors is empty
    else:
        # If empty pass on
        xc_ohe = xc

    # Return one hot encoded categorical predictors ---------------------------
    return xc_ohe


def split_trn_tst(df, i_trn, i_tst):
    """
    Split dataframe in training and testing dataframes.

    Parameters
    ----------
    df : dataframe
        Dataframe holding the data to split.
    i_trn : numpy array
        Array with indices of training data.
    i_tst : numpy array
        Array with indices of testing data.

    Returns
    -------
    df_trn : dataframe
        Dataframe holding the training data.
    df_tst : dataframe
         Dataframe holding the testing data.

    """
    # If dataframe is not empty
    if not df.empty:
        # Make split
        df_trn = df.iloc[i_trn].reset_index(drop=True)
        # Make split
        df_tst = df.iloc[i_tst].reset_index(drop=True)
    # If dataframe is empty
    else:
        # Make empty dataframes
        df_trn, df_tst = pd.DataFrame(), pd.DataFrame()

    # Return train test dataframes --------------------------------------------
    return df_trn, df_tst


def scale_x(x_trn, x_tst):
    """
    Scale x to mean = 0 and std = 1.

    Parameters
    ----------
    x_trn : dataframe
        Dataframe holding the training data.
    x_tst : dataframe
        Dataframe holding the testing data.

    Returns
    -------
    x_trn_sc : dataframe
        Dataframe holding the scaled training data.
    x_tst_sc : dataframe
        Dataframe holding the scaled testing data.
    x_scaler : sklearn scaler object
        Scaler object.

    """
    # Instantiate standard scaler
    x_scaler = StandardScaler(copy=True,
                              with_mean=True,
                              with_std=True)
    # Fit scaler and transform training data
    x_trn_sc = pd.DataFrame(x_scaler.fit_transform(x_trn),
                            columns=x_scaler.get_feature_names_out())
    # Transform testing data
    x_tst_sc = pd.DataFrame(x_scaler.transform(x_tst),
                            columns=x_scaler.get_feature_names_out())

    # Return scaled data and scaler object ------------------------------------
    return x_trn_sc, x_tst_sc, x_scaler


def get_class_w(y):
    """
    Compute class weights over array by counting occurrences.

    Parameters
    ----------
    y : ndarray
        Array containing class labels.

    Returns
    -------
    class_weights : dictionary
        Dictionary of class weights with class labels as keys.

    """
    # Count unique classes occurances
    counter = Counter(np.squeeze(y))
    # Compute total occurances
    total_class = sum(counter.values())

    # Return class weights as dictionary --------------------------------------
    return {key: 1-(count/total_class) for key, count in counter.items()}


def prep_predictor(task, y_trn):
    """
    Prepare a predictor with defined paramters.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    y : ndarray
        True values.

    Returns
    -------
    predictor : scikit-learn compatible predictor
        Prepared predictor object.

    """
    # If regression -----------------------------------------------------------
    if task['kind'] == 'reg':
        # Check predictor name
        if task['predictor_name'] == 'EN':
            # Predictor
            predictor = ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                fit_intercept=True,
                precompute=False,
                max_iter=1e4,
                copy_X=True,
                tol=1e-4,
                warm_start=True,
                positive=False,
                random_state=None,
                selection='random')
        # Check predictor name
        elif task['predictor_name'] == 'ET':
            # Predictor
            predictor = ExtraTreesRegressor(
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None)
        # Check predictor name
        elif task['predictor_name'] == 'RF':
            # Predictor
            predictor = RandomForestRegressor(
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None)
        # Check predictor name
        elif task['predictor_name'] == 'GB':
            # Predictor
            predictor = LGBMRegressor(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=100,
                subsample_for_bin=200000,
                objective=None,
                class_weight=None,
                min_split_gain=0.0,
                min_child_weight=1e-4,
                min_child_samples=1,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1.0,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=None,
                n_jobs=1,
                importance_type='gain')

    # If multi target regression ----------------------------------------------
    elif task['kind'] == 'reg_multi':
        # Check predictor name
        if task['predictor_name'] == 'EN':
            # Predictor
            predictor = MultiTaskElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                fit_intercept=True,
                copy_X=True,
                max_iter=1e4,
                tol=1e-4,
                warm_start=True,
                random_state=None,
                selection='random')
        # Check predictor name
        elif task['predictor_name'] == 'ET':
            # Predictor
            predictor = ExtraTreesRegressor(
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None)
        # Check predictor name
        elif task['predictor_name'] == 'RF':
            # Predictor
            predictor = RandomForestRegressor(
                n_estimators=100,
                criterion='friedman_mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None)

    # If classification -------------------------------------------------------
    elif task['kind'] == 'clf':
        # Check predictor name
        if task['predictor_name'] == 'LR':
            # Predictor
            predictor = LogisticRegression(
                penalty='elasticnet',
                tol=1e-4,
                C=1.0,
                fit_intercept=True,
                class_weight=get_class_w(y_trn),
                random_state=None,
                solver='saga',
                max_iter=1e4,
                multi_class='multinomial',
                verbose=0,
                warm_start=True,
                n_jobs=1,
                l1_ratio=0.5)
        # Check predictor name
        elif task['predictor_name'] == 'ET':
            # Predictor
            predictor = ExtraTreesClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='sqrt',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=get_class_w(y_trn),
                ccp_alpha=0.0,
                max_samples=None)
        # Check predictor name
        elif task['predictor_name'] == 'RF':
            # Predictor
            predictor = RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='sqrt',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=get_class_w(y_trn),
                ccp_alpha=0.0,
                max_samples=None)
        # Check predictor name
        elif task['predictor_name'] == 'GB':
            # Predictor
            predictor = LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=100,
                subsample_for_bin=200000,
                objective=None,
                class_weight=get_class_w(y_trn),
                min_split_gain=0.0,
                min_child_weight=1e-4,
                min_child_samples=1,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1.0,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=None,
                n_jobs=1,
                importance_type='gain')

    # Return the predictor ----------------------------------------------------
    return predictor


def prep_hyper_space(task):
    """
    Prepare hyperparameter space for hyperparamter tuning.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.

    Returns
    -------
    hyper_space : scikit-optimize hyperspace object
        Scikit-optimize hyperspace to search through for the optimal hyper
        paramters combination.

    """
    # If regression -----------------------------------------------------------
    if task['kind'] == 'reg':
        # Check predictor name
        if task['predictor_name'] == 'EN':
            # Hyper parameters search space
            hyper_space = {
                'alpha': Real(
                    1e-4, 1e4, prior='log-uniform', name='alpha'),
                'l1_ratio': Real(
                    1.1e-2, 1e-0, prior='uniform', name='l1_ratio')
            }
        # Check predictor name
        elif task['predictor_name'] == 'ET' or task['predictor_name'] == 'RF':
            # Hyper parameters search space
            hyper_space = {
                'n_estimators': Integer(
                    1, 500, prior='log-uniform', name='n_estimators'),
                'min_samples_split': Integer(
                    2, 500, prior='log-uniform', name='min_samples_split'),
                'min_samples_leaf': Integer(
                    1, 500, prior='log-uniform', name='min_samples_leaf'),
                'max_features': Real(
                    1e-2, 1e0, prior='uniform', name='max_features')
            }
        # Check predictor name
        elif task['predictor_name'] == 'GB':
            # Hyper parameters search space
            hyper_space = {
                'n_estimators': Integer(
                    1, 500, prior='log-uniform', name='n_estimators'),
                'num_leaves': Integer(
                    2, 500, prior='log-uniform', name='num_leaves'),
                'min_child_samples': Integer(
                    1, 500, prior='log-uniform', name='min_child_samples'),
                'colsample_bytree': Real(
                    1e-2, 1e0, prior='uniform', name='colsample_bytree')
            }

    # If multi target regression ----------------------------------------------
    elif task['kind'] == 'reg_multi':
        # Check predictor name
        if task['predictor_name'] == 'EN':
            # Hyper parameters search space
            hyper_space = {
                'alpha': Real(
                    1e-4, 1e4, prior='log-uniform', name='alpha'),
                'l1_ratio': Real(
                    1.1e-2, 1e-0, prior='uniform', name='l1_ratio')
            }
        # Check predictor name
        elif task['predictor_name'] == 'ET' or task['predictor_name'] == 'RF':
            # Hyper parameters search space
            hyper_space = {
                'n_estimators': Integer(
                    1, 500, prior='log-uniform', name='n_estimators'),
                'min_samples_split': Integer(
                    2, 500, prior='log-uniform', name='min_samples_split'),
                'min_samples_leaf': Integer(
                    1, 500, prior='log-uniform', name='min_samples_leaf'),
                'max_features': Real(
                    1e-2, 1e0, prior='uniform', name='max_features')
            }

    # If classification -------------------------------------------------------
    elif task['kind'] == 'clf':
        # Check predictor name
        if task['predictor_name'] == 'LR':
            # Hyper parameters search space
            hyper_space = {
                'C': Real(
                    1e-4, 1e4, prior='log-uniform', name='C'),
                'l1_ratio': Real(
                    1.1e-2, 1e-0, prior='uniform', name='l1_ratio')
            }
        # Check predictor name
        elif task['predictor_name'] == 'ET' or task['predictor_name'] == 'RF':
            # Hyper parameters search space
            hyper_space = {
                'n_estimators': Integer(
                    1, 500, prior='log-uniform', name='n_estimators'),
                'min_samples_split': Integer(
                    2, 500, prior='log-uniform', name='min_samples_split'),
                'min_samples_leaf': Integer(
                    1, 500, prior='log-uniform', name='min_samples_leaf'),
                'max_features': Real(
                    1e-2, 1e0, prior='uniform', name='max_features')
            }
        # Check predictor name
        elif task['predictor_name'] == 'GB':
            # Hyper parameters search space
            hyper_space = {
                'n_estimators': Integer(
                    1, 500, prior='log-uniform', name='n_estimators'),
                'num_leaves': Integer(
                    2, 500, prior='log-uniform', name='num_leaves'),
                'min_child_samples': Integer(
                    1, 500, prior='log-uniform', name='min_child_samples'),
                'colsample_bytree': Real(
                    1e-2, 1e0, prior='uniform', name='colsample_bytree')
            }

    # Return hyperspace -------------------------------------------------------
    return hyper_space


def accuracy_sample_weights_score(y_true, y_pred, class_weights):
    """
    Computes accuracy score weighted by the inverse if the frequency of a
    class.

    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
    class_weights : dictionary
        Class weights as inverse of frequency of class.

    Returns
    -------
    accuracy : float
        Prediction accuracy.

    """
    # Make y_true dataframe
    y_true_df = np.squeeze(y_true)
    # Make sample weights dataframe
    w = y_true_df.map(class_weights).values

    # Return sample weighted accuracy -----------------------------------------
    return accuracy_score(y_true, y_pred, sample_weight=w)


def print_hyper_param(task, i_cv, hp_params, hp_score):
    """
    Print best hyper paramters and related score to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        current cv repetition.
    hp_params : dictionary
        Best hyper params found.
    hp_score : dictionary
        Score for best hyper params found.

    Returns
    -------
    None.

    """
    # If regression
    if task['kind'] == 'reg' or task['kind'] == 'reg_multi':
        # Print data set
        print('Dataset: '+task['path_to_data'])
        # Print general information
        print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
              str('n rep outer cv: ')+str(task['n_rep_outer_cv'])+' | ' +
              str('n rep inner cv: ')+str(task['n_rep_inner_cv'])+' | ' +
              str(task['predictor_name']))
        # Print best hyperparameter and related score for regression task
        print('Best R2: '+str(np.round(hp_score, decimals=4))+' | ' +
              str(hp_params))
    # If classification
    elif task['kind'] == 'clf':
        # Print data set
        print('Dataset: '+task['path_to_data'])
        # Print general information
        print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
              str('n rep outer cv: ')+str(task['n_rep_outer_cv'])+' | ' +
              str('n rep inner cv: ')+str(task['n_rep_inner_cv'])+' | ' +
              str(task['predictor_name']))
        # Print best hyperparameter and related score for classification task
        print('Acc: '+str(np.round(hp_score, decimals=4))+' | ',
              str(hp_params))


def search_hyperparmameters(task, i_cv, predictor, hyper_space, g_cv_trn,
                            x_trn, y_trn, fit_params):
    """
    Inner loop of the nested cross-validation. Run a Bayesian search for best
    hyper paramter of predictor.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    predictor : sklearn predictor object
        EN, LR, ET, RF, GB preditor.
    hyper_space : scikit optimize hyper space object
        Hyper space that should be searched for optimale parameters.
    g_cv_trn : ndarray (n_samples)
        Group data.
    x_trn : ndarray (n_features x n_samples)
        Predictor train data.
    y_trn : ndarray (n_samples)
        Target train data.
    fit_params : dict
        Key value of fit params

    Returns
    -------
    Bayes search best predictor: sklearn compatible predictor
        Optimale predictor found by the Bayesian search.


    """
    # Prepare inner cross-validation scorer -----------------------------------
    # If regression use R²
    if (task['kind'] == 'reg' or task['kind'] == 'reg_multi'):
        # R2 score for regression
        scorer = 'r2'
    # classification use weighted accuracy
    elif task['kind'] == 'clf':
        # Weighted accuracy for classification
        scorer = make_scorer(accuracy_sample_weights_score,
                             greater_is_better=True,
                             **{'class_weights': get_class_w(y_trn)})

    # Get number of repetitions for inner cross-validation loop ---------------
    n_rep_inner_cv = get_n_rep_cv(task['n_rep_inner_cv'],
                                  task['test_size_frac'],
                                  g_cv_trn, 10)
    # Store number of inner cv repetitions in task
    task['n_rep_inner_cv'] = n_rep_inner_cv

    # Bayes search cross-validation -------------------------------------------
    # Instanciate inner cv for bayesian hyper parameter search
    bayes_search = BayesSearchCV(
        estimator=predictor,
        search_spaces=hyper_space,
        n_iter=task['n_rep_bayes'],
        optimizer_kwargs={'n_initial_points': int(2*task['n_rep_bayes']/3)},
        scoring=scorer,
        n_jobs=task['n_jobs'],
        n_points=5,
        refit=True,
        cv=GroupShuffleSplit(n_splits=task['n_rep_inner_cv'],
                             test_size=task['test_size_frac']),
        verbose=-1,
        pre_dispatch='2*n_jobs',
        random_state=None,
        error_score=0,
        return_train_score=False)
    # Bayesian search for best hyperparameter
    bayes_search.fit(x_trn, y=y_trn.squeeze(), groups=g_cv_trn, callback=None,
                     **fit_params)

    # Plot Bayes search results -----------------------------------------------
    # Plot and show optimizer results
    plot_convergence(bayes_search.optimizer_results_)
    plt.show()
    plot_evaluations(bayes_search.optimizer_results_[0])
    plt.show()
    plot_objective(bayes_search.optimizer_results_[0])
    plt.show()
    # Print best hyper paramters to console
    print_hyper_param(task, i_cv, bayes_search.best_params_,
                      bayes_search.best_score_)

    # Return best hyper params and related score ------------------------------
    return bayes_search.best_estimator_


def score_predictions(task, y_tst, y_pred, y):
    """
    Compute scores for predictions based on task.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    y_tst : ndarray
        Target test data.
    y_pred : ndarray
        Predicted values.
    y : ndarray
        All available data to compute true class weights for scoring.

    Returns
    -------
    results : tuple
        Returns scoring results. MAE, MSE and R² if task is regression.
        ACC and true class weights if task is classification.

    """
    # If regression
    if (task['kind'] == 'reg' or task['kind'] == 'reg_multi'):
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred, multioutput='raw_values')
        # Score predictions in terms of mse
        mse = mean_squared_error(y_tst, y_pred, multioutput='raw_values')
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred, multioutput='raw_values')
        # Results
        results = (mae, mse, r2)
    # If classification
    elif task['kind'] == 'clf':
        # Get class weights
        class_weights = get_class_w(y)
        # Calculate model fit in terms of acc
        acc = [accuracy_sample_weights_score(y_tst, y_pred, class_weights)]
        # Results
        results = (acc, class_weights)

    # Return results tuple ----------------------------------------------------
    return results


def get_model_importances(task, predictor):
    """
    Get model based feature importance.
    ref: Molnar, Christoph. "Interpretable machine learning. A Guide for
    Making Black Box Models Explainable", 2019.
    https://christophm.github.io/interpretable-ml-book/.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    predictor : sklearn predictor object
        EN, LR, ET, RF, GB preditor.

    Returns
    -------
    m_imp : ndarray
        Model based feature importance.

    """
    # If elastic net
    if task['predictor_name'] == 'EN':
        # Get model based predictor importance
        m_imp = np.squeeze(predictor.coef_.copy())
    # If logistic regression
    elif task['predictor_name'] == 'LR':
        # Get model based predictor importance
        m_imp = np.squeeze(predictor.coef_.copy())
    # If Extra trees or Random Forest
    elif task['predictor_name'] == 'ET' or task['predictor_name'] == 'RF':
        # Get model based predictor importance
        m_imp = np.sum([tree.tree_.compute_feature_importances(
            normalize=False) for tree in predictor.estimators_], axis=0)
    # If Gradient boosting
    elif task['predictor_name'] == 'GB':
        # Get model based predictor importance
        m_imp = predictor.feature_importances_
    # Other
    else:
        # Get model based predictor importance
        m_imp = []

    # Return model based predictor importances --------------------------------
    return m_imp


def get_permutation_importances(task, predictor, x_tst, y_tst, y, n_jobs=1):
    """
    Get permutation based feature importance.
    ref: Molnar, Christoph. "Interpretable machine learning. A Guide for
    Making Black Box Models Explainable", 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Breiman, Leo.“Random Forests.” Machine Learning 45 (1).
    Springer: 5-32 (2001).
    Ref: Fisher, Aaron, Cynthia Rudin, and Francesca Dominici.
    “All Models are Wrong, but Many are Useful: Learning a Variable’s
    Importance by Studying an Entire Class of Prediction Models
    Simultaneously.” http://arxiv.org/abs/1801.01489 (2018).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    predictor : sklearn predictor object
        EN, LR, ET, RF, GB preditor.
    x_tst : ndarray (n_features x n_samples)
        Predictor test data.
    y_tst : ndarray (n_samples)
        Target test data.
    y : ndarray (n_samples)
        Target data.
    n_jobs : integer, optional
        Number of parallell jobs. The default is 1.

    Returns
    -------
    p_imp : ndarray
        Permutation based feature importance.

    """
    # Prepare scorer ----------------------------------------------------------
    # If regression
    if task['kind'] == 'reg' or task['kind'] == 'reg_multi':
        # R2 score for regression
        scorer = 'r2'
    # If classification
    elif 'clf' in task['kind']:
        # Weighted accuracy for classification
        scorer = make_scorer(accuracy_sample_weights_score,
                             greater_is_better=True,
                             **{'class_weights': get_class_w(y)})

    # Permutation based predictor importance ----------------------------------
    # Get permutation based predictor importance
    p_imp = np.transpose(permutation_importance(
        predictor,
        x_tst,
        y_tst,
        scoring=scorer,
        n_repeats=task['n_rep_p_imp'],
        n_jobs=n_jobs,
        random_state=None)['importances_mean'])

    # Return permutation based predictor importance ---------------------------
    return p_imp


def get_shap_importances(task, predictor, x_tst):
    """
    Get SHAP (SHapley Additive exPlainations) based feature importance.
    ref: Molnar, Christoph. "Interpretable machine learning. A Guide for
    Making Black Box Models Explainable", 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Lundberg, Scott M., and Su-In Lee. “A unified approach to
    interpreting model predictions.” Advances in Neural Information Processing
    Systems. 2017.
    Ref: Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. “Consistent
    individualized feature attribution for tree ensembles.” arXiv preprint
    arXiv:1802.03888 (2018).
    Ref: Sundararajan, Mukund, and Amir Najmi. “The many Shapley values for
    model explanation.” arXiv preprint arXiv:1908.08474 (2019).
    Ref: Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. “Feature
    relevance quantification in explainable AI: A causality problem.” arXiv
    preprint arXiv:1910.13413 (2019).
    Ref: Slack, Dylan, et al. “Fooling lime and shap: Adversarial attacks on
    post hoc explanation methods.” Proceedings of the AAAI/ACM Conference on
    AI, Ethics, and Society. 2020.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    predictor : sklearn predictor object
        EN, LR, ET, RF, GB preditor.
    x_tst : ndarray (n_features x n_samples)
        Test data for shap computation.

    Returns
    -------
    x_tst_shap : dataframe
        Dataframe holding the samples used for shap computation.
    s_imp : ndarray
        SHAP based feature importance.

    """
    # Subsample test data
    x_tst_shap = x_tst.sample(
        n=min(x_tst.shape[0], task['max_samples_SHAP']),
        random_state=3141592,
        ignore_index=True)
    # If elastic net or logistic regression
    if (task['predictor_name'] == 'EN' or task['predictor_name'] == 'LR'):
        # Instanciate masker
        masker = shap.maskers.Partition(
            x_tst,
            max_samples=x_tst.shape[0],
            clustering='correlation')
        # Get shap based predictor importance
        s_imp = shap.explainers.Linear(predictor, masker).shap_values(
            x_tst_shap)
    # If tree based predictor
    elif (task['predictor_name'] == 'ET' or
          task['predictor_name'] == 'RF' or
          task['predictor_name'] == 'GB'):
        # Get shap based predictor importance
        s_imp = shap.explainers.Tree(predictor).shap_values(
            x_tst_shap, check_additivity=False)
    # Other
    else:
        # Get shap based predictor importance
        s_imp = []

    # Return shap based predictor importances and x_tst_shap ------------------
    return s_imp, x_tst_shap


def get_dependences(task, predictor, x_tst, scal, n_jobs=1):
    """
    Partial dependence of features. Partial dependence of a feature (or a set
    of features) corresponds to the average response of an estimator for each
    possible value of the feature.
    ref: Molnar, Christoph. "Interpretable machine learning. A Guide for
    Making Black Box Models Explainable", 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Friedman, Jerome H. “Greedy function approximation: A gradient
    boosting machine.” Annals of statistics (2001): 1189-1232.
    Ref: Zhao, Qingyuan, and Trevor Hastie. “Causal interpretations of
    black-box models.” Journal of Business & Economic Statistics, to appear.
    (2017).

    Parameters
    ----------
    predictor : sklearn predictor object
        EN, LR, ET, RF, GB preditor.
    x_tst : ndarray (n_features x n_samples)
        Predictor test data.
    scal : sklearn scaler object or None
        Std scaler object or None.
    n_jobs : integer, optional
        Number of parallell jobs. The default is 1.

    Returns
    -------
    dependence : ndarray
        The predictions for all the points in the grid, averaged over all
        samples in x_test_sc.
    grid : ndarray
        The values with which the grid has been created.
    dependence_inter : ndarray
        The predictions for all the interaction points in the grid, averaged
        over all samples in x_test_sc.
    grid_inter : ndarray
        The values with which the interaction grid has been created.

    """
    # Subsample test data -----------------------------------------------------
    x_tst_dep = x_tst.sample(
        n=min(x_tst.shape[0], task['max_samples_dep']),
        random_state=3141592,
        ignore_index=True)
    # Single predictor dependence ---------------------------------------------
    # Get indices of predictors
    x_ind = list(range(x_tst.columns.shape[0]))
    # Obtain predictors dependences in paralell
    dependence_results = (Parallel(n_jobs=n_jobs)(delayed(
        partial_dependence)(predictor, x_tst_dep, (i_x),
                            response_method='auto',
                            percentiles=(0.0, 1.0),
                            grid_resolution=20,
                            method='auto',
                            kind='average')
        for i_x in x_ind))
    # Unpack dependences
    dependence = [np.array(np.squeeze(c_res['average']), ndmin=1) for c_res in
                  dependence_results]
    # Unpack grid
    grid_scaled = [c_res['values'][0] for c_res in dependence_results]
    # Scale back grid
    grid = []
    # Iterate over grids
    for i_x, c_grid in enumerate(grid_scaled):
        # If scaler is present
        if scal is not None:
            # Scale back grid and append
            grid.append((c_grid * scal.scale_[i_x] + scal.mean_[i_x]).round(
                decimals=4))
        # If scaler is not present
        else:
            # Append grid
            grid.append(c_grid.round(decimals=4))

    # Interaction predictor dependence ----------------------------------------
    # Get indices for interactions
    x_ind_inter = list(combinations(x_ind, 2))
    # Obtain predictors dependences in paralell
    dependence_results_inter = (Parallel(n_jobs=n_jobs)(delayed(
        partial_dependence)(predictor, x_tst_dep, (i_x),
                            response_method='auto',
                            percentiles=(0.0, 1.0),
                            grid_resolution=10,
                            method='auto',
                            kind='average')
        for i_x in x_ind_inter))
    # Unpack dependences
    dependence_inter = [np.array(np.squeeze(c_res['average']), ndmin=1) for
                        c_res in dependence_results_inter]
    # Unpack grid
    grid_scaled_inter = [c_res['values'] for c_res in dependence_results_inter]
    # Scale back grid
    grid_inter = []
    # Iterate over grids
    for i_x, c_grid in enumerate(grid_scaled_inter):
        # If scaler is present
        if scal is not None:
            # Scale back
            grid_inter.append((
                (c_grid[0] * scal.scale_[x_ind_inter[i_x][0]] +
                 scal.mean_[x_ind_inter[i_x][0]]).round(decimals=4),
                (c_grid[1] * scal.scale_[x_ind_inter[i_x][1]] +
                 scal.mean_[x_ind_inter[i_x][1]]).round(decimals=4)))
        # If scaler is not present
        else:
            # Append grid
            grid_inter.append((c_grid[0].round(decimals=4),
                               c_grid[1].round(decimals=4)))

    # Return dependency and grids ---------------------------------------------
    return dependence, grid, dependence_inter, grid_inter


def s2p(path_save, variable):
    """
    Save variable as pickle file at path.

    Parameters
    ----------
    path_save : string
        Path ro save variable.
    variable : string
        Variable to save.

    Returns
    -------
    None.

    """
    # Save variable as pickle file
    with open(path_save, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(variable, filehandle)


def cross_validation(task, g_cv, y, xn, xc):
    """
    Outer loop of the nested cross-validation. Saves results to pickle file
    in path_to_results directory.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    y : dataframe
        Target dataframe.
    xn : dataframe
        Numerical predictors dataframe.
    xc : dataframe
        Categorical predictors dataframe.

    Returns
    -------
    None.

    """
    # One hot encode ----------------------------------------------------------
    # If one hot encoding is necessary for predictors
    if (task['predictor_name'] == 'EN' or
        task['predictor_name'] == 'LR' or
        task['predictor_name'] == 'ET' or
            task['predictor_name'] == 'RF'):
        # One hot encode categorical predictors
        xc_ohe = ohe(xc)
        # Concatenate numerical and encoded categorical predictors
        x = pd.concat([xn, xc_ohe], axis=1)
    # If one hot encoding is not necessary for predictors
    else:
        # Concatenate numerical and categorical predictors
        x = pd.concat([xn, xc], axis=1)
    # Store column names
    task['x_names'] = list(x.columns)

    # Initialize results lists ------------------------------------------------
    # Initialize true values list
    true_values = []
    # Initialize predicted values list
    pred_values = []
    # Initialize score list
    scores = []
    # Initialize model based feature importance list
    m_importances = []
    # Initialize permutation based feature importance list
    p_importances = []
    # Initialize SHAP based feature importance list
    s_importances = []
    # Initialize dependences list of list
    dependences = []
    # Initialize shuffle data score list
    scores_sh = []
    # Initialize shuffle data model based feature importance list
    m_importances_sh = []
    # Initialize shuffle data permutation based feature importance list
    p_importances_sh = []
    # Initialize shuffle data SHAP based feature importance list
    s_importances_sh = []

    # Main cross-validation loop ----------------------------------------------
    # Instanciate main cv splitter with fixed random state for comparison
    cv = GroupShuffleSplit(
        n_splits=task['n_rep_outer_cv'],
        test_size=task['test_size_frac'],
        random_state=3141592)
    # Loop over main (outer) cross validation splits
    for i_cv, (i_trn, i_tst) in enumerate(cv.split(g_cv, groups=g_cv)):
        # Save loop start time
        t_start = time()

        # Split in training and testing set -----------------------------------
        # Split groups
        g_cv_trn, g_tst = split_trn_tst(g_cv, i_trn, i_tst)
        # Split targets
        y_trn, y_tst = split_trn_tst(y, i_trn, i_tst)
        # Split predictors
        x_trn, x_tst = split_trn_tst(x, i_trn, i_tst)

        # Scaling -------------------------------------------------------------
        # If scaling is necessary for predictor
        if (task['predictor_name'] == 'EN' or task['predictor_name'] == 'LR'):
            # Scale predictors
            x_trn, x_tst, x_scaler = scale_x(x_trn, x_tst)
        # If scaling is not necessary for predictor
        else:
            x_scaler = None

        # Prepare predictor ---------------------------------------------------
        # Get prepared predictor
        predictor = prep_predictor(task, y_trn)

        # Prepare fit params --------------------------------------------------
        # If predictor is Gradient Boosting and xc_names not empty
        if task['predictor_name'] == 'GB':
            # Fit params for GB
            fit_params = {'feature_name': task['x_names'],
                          'categorical_feature': task['xc_names']}
        # If other predictor
        else:
            # Fit params for other predictor
            fit_params = {}

        # Search for best hyper parameters and fit predictor ------------------
        # Prepare hyper parameter space
        hyper_space = prep_hyper_space(task)
        # Search best hyper parameters and return best predictor
        predictor = search_hyperparmameters(task, i_cv, predictor, hyper_space,
                                            g_cv_trn, x_trn, y_trn, fit_params)

        # Analyse predictor ---------------------------------------------------
        # Predict testing samples
        y_pred = predictor.predict(x_tst)
        # Append true values
        true_values.append(np.squeeze(y_tst.to_numpy()))
        # Append predictions
        pred_values.append(y_pred)
        # Score predictions
        scores.append(score_predictions(task, y_tst, y_pred, y))
        # Model predictor importances
        m_importances.append(get_model_importances(task, predictor))
        # Permutation predictor importances
        p_importances.append(get_permutation_importances(
            task, predictor, x_tst, y_tst, y, n_jobs=task['n_jobs']))
        # SHAP predictor importances
        s_importances.append(get_shap_importances(task, predictor, x_tst))
        # Predictor dependence
        dependences.append(get_dependences(task, predictor, x_tst, x_scaler,
                                           n_jobs=task['n_jobs']))

        # Prepare and fit shuffle data predictor ------------------------------
        # Clone predictor
        predictor_sh = clone(predictor)
        # Shuffle training targets
        y_trn_sh = shuffle(y_trn)
        # Refit predictor with shuffled targets
        predictor_sh.fit(x_trn, np.squeeze(y_trn_sh), **fit_params)

        # Analyse shuffle predictor -------------------------------------------
        # Predict testing samples
        y_pred_sh = predictor_sh.predict(x_tst)
        # Score predictions
        scores_sh.append(score_predictions(task, y_tst, y_pred_sh, y))
        # Model predictor importances
        m_importances_sh.append(get_model_importances(task, predictor_sh))
        # Permutation predictor importances
        p_importances_sh.append(get_permutation_importances(
            task, predictor_sh, x_tst, y_tst, y, n_jobs=task['n_jobs']))
        # SHAP predictor importances
        s_importances_sh.append(get_shap_importances(
            task, predictor_sh, s_importances[-1][1]))

        # Compile and save intermediate results and task ----------------------
        # Create results
        results = {
            'true_values': true_values,
            'pred_values': pred_values,
            'scores': scores,
            'm_importances': m_importances,
            'p_importances': p_importances,
            's_importances': s_importances,
            'dependences': dependences,
            'scores_sh': scores_sh,
            'm_importances_sh': m_importances_sh,
            'p_importances_sh': p_importances_sh,
            's_importances_sh': s_importances_sh
        }
        # Save results as pickle file
        s2p(task['path_to_results']+'/results_'+'_'.join(task['y_name']) +
            '.pickle', results)
        # Save task as pickle file
        s2p(task['path_to_results']+'/task_'+'_'.join(task['y_name']) +
            '.pickle', task)

        # Display intermediate results ----------------------------------------
        if task['kind'] == 'reg':
            # Print current R2
            print('Current CV loop R2: '+str(np.round(
                scores[-1][2], decimals=4)))
            # Print running mean R2
            print('Running mean R2: '+str(np.round(
                np.mean([i[2] for i in scores]), decimals=4)))
            # Print running mean shuffle R2
            print('Running shuffle mean R2: '+str(np.round(
                np.mean([i[2] for i in scores_sh]), decimals=4)))
            # Print elapsed time
            print('Elapsed time: '+str(np.round(
                time() - t_start, decimals=1)), end='\n\n')
        elif task['kind'] == 'reg_multi':
            # Print current R2
            print('Current CV loop R2: '+str(np.round(
                scores[-1][2], decimals=4)))
            # Print running mean R2
            print('Running mean R2: '+str(np.round(
                np.mean([i[2] for i in scores], axis=0), decimals=4)))
            # Print running mean shuffle R2
            print('Running shuffle mean R2: '+str(np.round(
                np.mean([i[2] for i in scores_sh], axis=0), decimals=4)))
            # Print elapsed time
            print('Elapsed time: '+str(np.round(
                time() - t_start, decimals=1)), end='\n\n')
        elif task['kind'] == 'clf':
            # Print current acc
            print('Current CV loop acc: '+str(np.round(
                scores[-1][0], decimals=4)))
            # Print running mean acc
            print('Running mean acc: '+str(np.round(
                np.mean([i[0] for i in scores]), decimals=4)))
            # Print running mean shuffle acc
            print('Running shuffle mean acc: '+str(np.round(
                np.mean([i[0] for i in scores_sh]), decimals=4)))
            # Print elapsed time
            print('Elapsed time: '+str(np.round(
                time() - t_start, decimals=1)), end='\n\n')


def main():
    """
    Main function of the machine-learning based data analysis.

    Returns
    -------
    None.

    """
    ###########################################################################
    # Specify analysis
    ###########################################################################

    # 1. Specify task ---------------------------------------------------------
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    n_jobs = -2
    # Number of outer CV repetitions. 'auto' or int (default: 'auto')
    # 'auto' adjusts number of repetitions so that number of predictions
    # approximately equals 10 times the number of available samples and
    # guarantees a minimum of 5 repetitions
    n_rep_outer_cv = 50
    # Test size fraction of groups in CV. float (]0,1], default: 0.2)
    test_size_frac = 0.2
    # Number of bayes optimization repetitions. int (default: 150).
    n_rep_bayes = 150
    # Number of inner CV repetitions. 'auto' or int (default: 'auto')
    # 'auto' adjusts number of repetitions so that number of predictions
    # approximately equals 10 times the number of available samples and
    # guarantees a minimum of 5 repetitions
    n_rep_inner_cv = 50
    # Specify predictor. string
    # if kind == clf: LR, ET, RF, GB -> classification
    # if kind == reg: EN, ET, RF, GB -> regression
    # if kind == reg_multi: EN, ET, RF -> multi target regression
    predictor_name = 'ET'
    # Number of permutation importance repetitions. int (default: 10).
    n_rep_p_imp = 10
    # Limit number of SHAP importance samples. int (default: 100).
    max_samples_SHAP = 100
    # Limit number of partial dependence samples. int (default: 1000).
    max_samples_dep = 1000

    # 2. Specify data ---------------------------------------------------------

    # Housing data - regression
    # Specify task kind. string (clf, reg, reg_multi)
    kind = 'reg'
    # Specify path to data. string
    # path_to_data = 'data/housing.csv'
    path_to_data = 'data/mindfulness.csv'

    # Specify grouping for CV split. list of string
    g_cv_name = ['sample_id']
    # Specify numerical predictor names. list of string or []
    # xn_names = ['longitude',
    #             'latitude',
    #             'housing_median_age',
    #             'total_rooms',
    #             'total_bedrooms',
    #             'population',
    #             'households',
    #             'median_income']
    
    xn_names = [
        "ffmq_ds1",
        "ffmq_aa1",
        "ffmq_ds2",
        "ffmq_aa2",
        "ffmq_nr1",
        "ffmq_nr2",
        "ffmq_aa3",
        "ffmq_nj1",
        "ffmq_ob1",
        "ffmq_aa4",
        "ffmq_nr3",
        "ffmq_ob2",
        "ffmq_nr4",
        "ffmq_nr5",
        "ffmq_nj2",
        "ffmq_ob3",
        "ffmq_ds3",
        "ffmq_nr6",
        "ffmq_nj3",
        "ffmq_ob4",
        "ffmq_ds4",
        "ffmq_nr7",
        "ffmq_nj4",
        "upps_ur_1",
        "upps_ur_2",
        "upps_ur_3",
        "upps_ur_4",
        "upps_ur_5",
        "upps_pm_1",
        "upps_pm_2",
        "upps_pm_3",
        "upps_pm_4",
        "upps_pm_5",
        "upps_pe_1",
        "upps_pe_2",
        "upps_pe_3",
        "upps_pe_4",
        "upps_pe_5",
        "upps_ss_1",
        "upps_ss_2",
        "upps_ss_3",
        "upps_ss_4",
        "upps_ss_5",
        "dmq_cope_1",
        "dmq_cope_2",
        "dmq_cope_3",
        "dmq_cope_4",
        "dmq_cope_5",
        "geschlecht_kod_male",
        "erwerbstaetig_sub"]

    # Specify categorical predictor names. list of string or []
    # xc_names = ['ocean_proximity']
    xc_names = []
    # Specify target name(s). list of strings or []
    # y_names = ['median_house_value']
    y_names = ['audit']
    # Specify index of rows to drop manually. list of int or []
    rows_to_drop = []

    # # Iris data - classification
    # # Specify task kind. string (clf, reg, reg_multi)
    # kind = 'clf'
    # # Specify path to data. string
    # path_to_data = 'data/iris_3class_20220307.xlsx'
    # # Specify sheet name. string
    # sheet_name = 'Sheet1'
    # # Specify grouping for CV split. list of string
    # g_cv_name = ['sample_id']
    # # Specify numerical predictor names. list of string or []
    # xn_names = ['sepal_length',
    #             'sepal_width',
    #             'petal_length',
    #             'petal_width']
    # # Specify categorical predictor names. list of string or []
    # xc_names = []
    # # Specify target name(s). list of strings or []
    # y_names = ['iris']
    # # Specify index of rows to drop manually
    # rows_to_drop = []

    # # Linnerud data - multi target regression
    # # Specify task kind. string (clf, reg, reg_multi)
    # kind = 'reg_multi'
    # # Specify path to data. string
    # path_to_data = 'data/linnerud_3target_20220307.xlsx'
    # # Specify sheet name. string
    # sheet_name = 'Sheet1'
    # # Specify grouping for CV split. list of string
    # g_cv_name = ['sample_id']
    # # Specify numerical predictor names. list of string or []
    # xn_names = ['chins',
    #             'situps',
    #             'jumps']
    # # Specify categorical predictor names. list of string or []
    # xc_names = []
    # # Specify target name(s). list of strings or []
    # y_names = ['weight',
    #            'waist',
    #            'pulse']
    # # Specify index of rows to drop manually. list of int or []
    # rows_to_drop = []

    # # Radon data - high cardinality predictor regression
    # # Specify task kind. string (clf, reg, reg_multi)
    # kind = 'reg'
    # # Specify path to data. string
    # path_to_data = 'data/radon_20220405.xlsx'
    # # Specify sheet name. string
    # sheet_name = 'Sheet1'
    # # Specify grouping for CV split. list of string
    # g_cv_name = ['sample_id']
    # # Specify numerical predictor names. list of string or []
    # xn_names = ['Uppm']
    # # Specify categorical predictor names. list of string or []
    # xc_names = ['county_code',
    #             'floor']
    # # Specify target name(s). list of strings or []
    # y_names = ['log_radon']
    # # Specify index of rows to drop manually. list of int or []
    # rows_to_drop = []

    ###########################################################################

    # Create main results directory -------------------------------------------
    # Create path to results directory
    path_to_results = ('res_'+kind+'_'+predictor_name+'_'+'_'.join(y_names))
    # Create results dir
    create_dir(path_to_results)

    # Load data ---------------------------------------------------------------
    df = pd.read_csv(path_to_data, sep='\t', skiprows=rows_to_drop, index_col=0).reset_index().rename(
        columns={"index": "sample_id"})
    # df = pd.read_csv(path_to_data, skiprows=rows_to_drop).reset_index().rename(
        # columns={"index": "sample_id"})
    df.sample_id += 1

    # Load groups from csv file
    g_cv = df[g_cv_name].astype('int64')
    # Load targets from csv file
    y = df[y_names].astype('float64')
    # Load numerical predictors from csv file
    xn = df[xn_names].astype('float64')
    # Load categorical predictors from csv file
    xc = df[xc_names]
    xc = ohe(xc)  # make this optional (only if not already encoded)
    xc_names = list(xc.columns)

    # Create task variable ----------------------------------------------------
    task = {
        'n_rep_outer_cv': n_rep_outer_cv,
        'n_jobs': n_jobs,
        'test_size_frac': test_size_frac,
        'n_rep_bayes': n_rep_bayes,
        'n_rep_inner_cv': n_rep_inner_cv,
        'kind': kind,
        'predictor_name': predictor_name,
        'n_rep_p_imp': n_rep_p_imp,
        'max_samples_SHAP': max_samples_SHAP,
        'max_samples_dep': max_samples_dep,
        'path_to_data': path_to_data,
        'g_cv_name': g_cv_name,
        'xn_names': xn_names,
        'xc_names': xc_names,
        'y_names': y_names,
        'path_to_results': path_to_results
    }

    # Get number of repetitions for outer cross-validation loop ---------------
    n_rep_outer_cv = get_n_rep_cv(n_rep_outer_cv, test_size_frac, g_cv, 10)
    # Store number of outer cv repetitions in task
    task['n_rep_outer_cv'] = n_rep_outer_cv

    # Cross-validation --------------------------------------------------------
    # If multi target regression
    if task['kind'] == 'reg_multi':
        # Add task index
        task['i_y'] = 0
        # Add targets names to task
        task['y_name'] = y_names
        # Drop rows with nans
        gi_cv, yi, xin, xic = drop_nan_rows(g_cv, y, xn, xc)
        # Run cross-validation
        cross_validation(task, gi_cv, yi, xin, xic)
    # If other
    else:
        # Iterate over y names
        for i_y, y_name in enumerate(y_names):
            # Add task index
            task['i_y'] = i_y
            # Add target name to task
            task['y_name'] = [y_name]
            # Get current target
            yi = y[y_name].to_frame()
            # Drop rows with nans
            gi_cv, yi, xin, xic = drop_nan_rows(g_cv, yi, xn, xc)
            # Run cross-validation
            cross_validation(task, gi_cv, yi, xin, xic)


if __name__ == "__main__":
    main()
