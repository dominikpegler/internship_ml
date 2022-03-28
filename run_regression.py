from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from get_data import get_mindfulness as get_data
from regressors import get_regressor
from utils import split_train_test
from skopt.plots import plot_objective, plot_evaluations, plot_convergence
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


def main():

    start = time.time()

    X, y = get_data("ffmq-overall")

    reg_type = "elasticnet"
    # for reg_type in ["elasticnet", "rf", "extratrees", "gradientboost"]:

    # "elasticnet", "rf", "extratrees", "gradientboost"
    reg, hyperparams_dist = get_regressor(reg_type)

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
            n_iter=200,  # 200
            cv=5,
            n_jobs=8,
            random_state=0
        )

        result = search_reg.fit(X_train, y_train.values.ravel())

        print(f"Split {i_cv}:", result.best_estimator_)
        print("train score:", round(result.score(X_train, y_train), 5))
        print("test  score:", round(result.score(X_test, y_test), 5))
        print("\n")

    # hp optimization plots
    plot_convergence(search_reg.optimizer_results_)
    plt.savefig("plot_convergence_"+reg_type+".png")
    plot_evaluations(search_reg.optimizer_results_[0])
    plt.savefig("plot_evaluations_"+reg_type+".png")
    plot_objective(search_reg.optimizer_results_[0])
    plt.savefig("plot_objective_"+reg_type+".png")

    # scatter + regression line plots
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.3)
    x = pd.DataFrame(np.linspace(0, 5), columns=X.columns)
    ax.plot(x.values, result.best_estimator_.predict(x))
    ax.title.set_text(reg_type)
    try:
        plt.text(
            0.4, 25, f"y = {result.best_estimator_.coef_[0].round(2)}x + {result.best_estimator_.intercept_.round(2)}", fontsize=12)
    except:
        ...
    plt.text(
        0.4, 22, f"Best score $R²$ = {result.best_score_.round(3)} ($r$ = {(result.best_score_.round(3)**(1/2)).round(2)})", fontsize=12)
    ax.set_ylim(0, 30)
    ax.set_xlim(0, 5)
    fig.savefig("predictions_"+reg_type+".png")

    print(f"Execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()
