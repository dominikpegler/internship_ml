from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV 
from get_data import get_mindfulness as get_data
from regressors import get_regressor
from utils import split_train_test
from skopt.plots import plot_objective, plot_evaluations, plot_convergence
import matplotlib.pyplot as plt
import numpy as np
import time


def main():

    start = time.time()
    

    X, y = get_data()


    reg_type = "rf"
    #for reg_type in ["elasticnet", "rf", "extratrees", "gradientboost"]:

    reg,hyperparams_dist = get_regressor(reg_type) # "elasticnet", "rf", "extratrees", "gradientboost"


    outer_cv = GroupShuffleSplit(n_splits=5,
                                 test_size=0.2,
                                 random_state=0
                                )

    # iterate over outer CV splitter
    for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y, groups=X.index), start=1):

        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)

        # nested CV with parameter optimization
        search_reg = BayesSearchCV(
            estimator=reg,
            search_spaces=hyperparams_dist,
            n_iter=200,
            cv=5,
            n_jobs=8,
            random_state=0
        )

        result = search_reg.fit(X_train, y_train.values.ravel())

        print(f"Split {i_cv}:", result.best_estimator_)
        print("train score:", round(result.score(X_train, y_train), 5))
        print("test  score:", round(result.score(X_test, y_test), 5))
        print("\n")

    plot_convergence(search_reg.optimizer_results_)
    plt.show()
    plot_evaluations(search_reg.optimizer_results_[0])
    plt.show()
    plot_objective(search_reg.optimizer_results_[0])
    plt.show()

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()