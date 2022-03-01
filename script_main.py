from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score  # alternative: cross_validate
from sklearn.linear_model import LinearRegression
from get_data import get_data
import time


def main():

    start = time.time()

    X, y = get_data()

    linreg_mod = LinearRegression()

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = cross_val_score(linreg_mod, X, y, cv=cv)

    print(f"{scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}")
    print(f"\nExecution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
