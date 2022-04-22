import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from skopt.space import Real, Integer


max_features = Real(1e-2,1e0,prior="uniform")
max_depth.append(None)
min_samples_split = Integer(2,256,prior="log-uniform")
min_samples_leaf = Integer(1,256,prior="log-uniform")
learning_rate = Real(1e-3,1e0,prior="log-uniform")
num_leaves = Integer(2,256,prior="log-uniform")
min_child_samples = Integer(1,256,prior="log-uniform")
n_estimators = 256


def get_regressor(reg_type="ElasticNet"):
    """
    scikit-learn regressors ready to use for bayesCV
        
        returns regressor model and dictionary for possible hyperparameter distributions
    
        Parameters
        ----------
        reg_type : {'ElasticNet',
                    'RandomForestRegressor',
                    'ExtraTreesRegressor',
                    'GradientBoostingRegressor',
                    'LGBMRegressor'} default='ElasticNet'
    
    """
    
    if reg_type == "ElasticNet":
        reg = ElasticNet()
        hyper_space = {
            "alpha": (1e-4,1e3,"log-uniform"),
            "l1_ratio": (1e-3,1.0,"uniform")
            }

    elif reg_type == "RandomForestRegressor":
        reg = RandomForestRegressor( n_estimators=n_estimators,random_state = 0)
        hyper_space = {
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }

    elif reg_type == "ExtraTreesRegressor":
        reg = ExtraTreesRegressor( n_estimators=n_estimators,random_state = 0)
        hyper_space = {
           'max_features': max_features,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
       }
        
        
    elif reg_type == "LGBMRegressor":
        reg = LGBMRegressor(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.1,
                n_estimators=n_estimators,
                subsample_for_bin=200000,
                objective=None,
                class_weight=None,
                min_split_gain=0.0,
                min_child_weight=1e-4,
                min_child_samples=0,
                subsample=1.0,
                subsample_freq=0,
                colsample_bytree=1e-2,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=None,
                n_jobs=1,
                importance_type='gain',
                )
        
        hyper_space = {
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
        }   
        
    else:
        raise Exception("No valid regressor defined!")
        
    return reg,hyper_space
