from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVR
from scipy.stats import loguniform
from get_data import get_mindfulness as get_data
from utils import split_train_test
import time


def main():

    start = time.time()

    X, y = get_data()

    loss = ["squared_epsilon_insensitive"]
    C = loguniform(1e-4,1e4)
    hyperparams_grid = {
        'loss': loss,
        'C': C}

    reg = LinearSVR(random_state=0,max_iter=1000,dual=False)

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
            estimator=reg, n_iter=1000, param_distributions=hyperparams_grid, cv=5)
        result = search_reg.fit(X_train, y_train.values.ravel())

        print(f"Split {i_cv}:", result.best_estimator_)
        print("train score:", round(result.score(X_train, y_train), 5))
        print("test  score:", round(result.score(X_test, y_test), 5))
        print("\n")

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
