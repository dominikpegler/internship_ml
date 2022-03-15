import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]


def get_regressor(reg_type="elasticnet"):
    """
    scikit-learn regressors ready to use for bayesCV
        
        returns regressor model and dictionary for possible hyperparameter distributions
    
        Parameters
        ----------
        reg_type : {'elasticnet', 'rf', 'extratrees', 'gradientboost'} default='elasticnet'
    
    """
    
    if reg_type == "elasticnet":
        reg = ElasticNet()
        hyperparams_dist = {
            "alpha": (1e-4,1e3,"log-uniform"),
            "l1_ratio": (1e-3,1.0,"uniform")
            }

    elif reg_type == "rf":
        reg = RandomForestRegressor(random_state = 0)
        hyperparams_dist = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

    elif reg_type == "extratrees":
        reg = ExtraTreesRegressor(random_state = 0)
        hyperparams_dist = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        
    elif reg_type == "gradientboost":
        reg = GradientBoostingRegressor(random_state=0)
        hyperparams_dist = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }        
    else:
        raise Exception("No valid regressor defined!")
        
    return reg,hyperparams_dist