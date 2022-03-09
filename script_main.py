from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from get_data import get_housing_data
from utils import split_train_test
import time


def main():

    start = time.time()

    X, y = get_housing_data()
    hyperparams_grid = {"alpha": [1, 10, 50, 100, 150, 175, 200, 225, 250, 275,
                                  300, 325, 350, 375, 400, 450, 500, 700, 1000],
                        "l1_ratio": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                     0.8, 0.9, 0.95, 0.99]}
    reg = ElasticNet()

    # outer CV
    outer_cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0)

    # Iterate over outer CV splitter
    for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y, groups=X.index), start=1):

        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)

        print(f"Split {i_cv}")

        # Nested CV with parameter optimization
        search_reg = RandomizedSearchCV(
            estimator=reg, n_iter=100, param_distributions=hyperparams_grid, cv=5)
        result = search_reg.fit(X_train, y_train)

        print("best model:", result.best_estimator_)
        print("score on train data:", result.score(X_train, y_train))
        print("score on  test data:", result.score(X_test, y_test))

        print("\n")

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
