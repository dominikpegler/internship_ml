from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from get_data import get_data
from utils import split_train_test
import time

def main():

    start = time.time()

    X, y = get_data()

    hps = [0.0, 1e1, 1e2, 1e3, 1e4]
    hp_name = 'alpha'

    # main cv splitter
    cv = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=0)

    # Iterate over main splitter
    for i_cv, (i_train, i_test) in enumerate(cv.split(X, y, groups=X.index), start=1):

        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)

        print(f"Split {i_cv}")

        # iterate over possible models
        for j, hp in enumerate(hps):
            m = Ridge(alpha=hp)
            result = m.fit(X_train, y_train)
            score = result.score(X_test, y_test)
            print(f"{hp_name}={hp}: R²={score:.2f}")

        print("\n")

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()

