import shap
import numpy as np

def get_feature_importance(reg_type, model, X_test,max_samples_SHAP=99):
    """
    Parameters
    -------
    reg_type: String
    model: Model type
    max_samples_SHAP: Integer (default 99)

    Returns
    -------
    Tuple
        (Modelbased importance, shap base importance).

    """

    # Get model based x importance
    if reg_type in ["ElasticNet"]:
        m_imp = np.squeeze(model.coef_.copy())
        
    elif reg_type in [
            "ExtraTreesRegressor", 
            "RandomForestRegressor",
            ]:
        m_imp = np.sum([tree.tree_.compute_feature_importances(
            normalize=False) for tree in model.estimators_], axis=0)
    
    elif reg_type in ["LGBMRegressor","GradientBoostingRegressor"]:
         m_imp = model.feature_importances_
    
    else:
        m_imp = list()
        
        
    # Subsample x_tst_sc
    x_tst_sc_shap = X_test.sample(
        n=min(X_test.shape[0], max_samples_SHAP),
        random_state=0,
        ignore_index=True)
    
    # Get shap based x importance
    if reg_type in ["ElasticNet"]:
        # Instanciate masker
        masker = shap.maskers.Partition(
            X_test,
            max_samples=X_test.shape[0],
            clustering='correlation')
        # Instantiate explainer and get shap values
        s_imp = shap.explainers.Linear(model, masker).shap_values(
            x_tst_sc_shap)
    
    elif reg_type in [
            "ExtraTreesRegressor",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "LGBMRegressor",
                      ]:
        # Instanciate explainer and get shap values
        s_imp = shap.explainers.Tree(model).shap_values(x_tst_sc_shap)
        
    return m_imp,s_imp