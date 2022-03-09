from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from get_data import get_data
from utils import split_train_test
import numpy as np
import time


def main():

    start = time.time()

    X, y = get_data()

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    hyperparams_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

    reg = RandomForestRegressor(random_state=0)

    # outer CV
    outer_cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0)

    # Iterate over outer CV splitter
    for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y, groups=X.index), start=1):

        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)

        # Nested CV with parameter optimization
        search_reg = RandomizedSearchCV(
            estimator=reg, n_iter=100, param_distributions=hyperparams_grid, cv=5)
        result = search_reg.fit(X_train, y_train.values.ravel())

        print(f"Split {i_cv}:", result.best_estimator_)
        print("train score:", round(result.score(X_train, y_train), 5))
        print("test  score:", round(result.score(X_test, y_test), 5))
        print("\n")

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
