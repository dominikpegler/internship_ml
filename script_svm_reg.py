from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from scipy.stats import loguniform
from get_data import get_mindfulness as get_data
from utils import split_train_test
import time


def main():

    start = time.time()

    X, y = get_data()

    kernel = ["rbf"]#, "poly", "rbf", "sigmoid"]
    degree = [1]
    gamma = ["auto","scale"] # loguniform(1e-6,1e0)
    C = loguniform(1e0,1e3)
    epsilon = loguniform(1e-3,1e1)
    shrinking = [True, False]

    hyperparams_grid = {
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,
        'C': C,
        'epsilon': epsilon,
        'shrinking': shrinking}

    reg = SVR()

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
