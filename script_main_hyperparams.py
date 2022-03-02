from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from get_data import get_data
import numpy as np
from utils import split_train_test
import time


def main():

    # Number of trees in Random Forest
    rf_n_estimators = [int(x) for x in np.linspace(200, 1000, 5)]
    rf_n_estimators.append(1500)
    rf_n_estimators.append(2000)

    # Maximum number of levels in tree
    rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
    # Add the default as a possible value
    rf_max_depth.append(None)

    # Number of features to consider at every split
    rf_max_features = ['auto', 'sqrt', 'log2']

    # Criterion to split on
    rf_criterion = ['squared_error', 'absolute_error']

    # Minimum number of samples required to split a node
    rf_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

    # Minimum decrease in impurity required for split to happen
    rf_min_impurity_decrease = [0.0, 0.05, 0.1]

    # Method of selecting samples for training each tree
    rf_bootstrap = [True, False]

    # Create the grid
    rf_grid = {'n_estimators': rf_n_estimators,
               'max_depth': rf_max_depth,
               'max_features': rf_max_features,
               'criterion': rf_criterion,
               'min_samples_split': rf_min_samples_split,
               'min_impurity_decrease': rf_min_impurity_decrease,
               'bootstrap': rf_bootstrap}

    start = time.time()

    dummy_hyperparams = [2, 5, 8]

    X, y = get_data()

    # main cv splitter
    cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0)

    # Iterate over main splitter
    for i_cv, (i_train, i_test) in enumerate(cv.split(X, y, groups=X.index), start=1):

        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)

        # Create the model to be tuned
        m = RandomForestRegressor()

        # Create the random search Random Forest
        m_search = RandomizedSearchCV(estimator=m, param_distributions=rf_grid,
                                      n_iter=200, cv=3, verbose=2, random_state=42,
                                      n_jobs=-1)

        # Fit the random search model
        m_search.fit(X_train, y_train.ravel())

        # View the best parameters from the random search
        print(f"Split: {i_cv}, Best Params: {m_search.best_params_}")

    print(f"\nExecution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
