from sklearn.model_selection import GroupShuffleSplit
#from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV # scikit-optimize
from sklearn.linear_model import ElasticNet
#from scipy.stats import uniform, loguniform
from get_data import get_mindfulness as get_data
from utils import split_train_test
import time


def main():

    start = time.time()
    
    X, y = get_data()
    
    hyperparams_dist = {
    #        "alpha": loguniform(1e-4,1e3),
    #        "l1_ratio": uniform(0,1)
        "alpha": (1e-4,1e3,"log-uniform"),
        "l1_ratio": (1e-3,1.0,"uniform")
    }
    
    reg = ElasticNet()
    
    outer_cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0)
    
    # iterate over outer CV splitter
    for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y, groups=X.index), start=1):
    
        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)
    
        # nested CV with parameter optimization
        #search_reg = RandomizedSearchCV(
        #    estimator=reg, n_iter=1000, param_distributions=hyperparams_dist, cv=5)
        search_reg = BayesSearchCV(
            estimator=reg,
            search_spaces=hyperparams_dist,
            n_iter=50,
            cv=5,
            n_jobs=8,
            random_state=0)
        result = search_reg.fit(X_train, y_train)
    
        print(f"Split {i_cv}:", result.best_estimator_)
        print("train score:", round(result.score(X_train, y_train), 5))
        print("test  score:", round(result.score(X_test, y_test), 5))
        print("\n")
    
    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
