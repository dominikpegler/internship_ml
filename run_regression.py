from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from get_data import get_mindfulness as get_data
from regressors import get_regressor
from utils import split_train_test
from skopt.plots import plot_objective, plot_evaluations, plot_convergence
import matplotlib.pyplot as plt
import time
import traceback
import csv
import os
import json as json
from feature_importance import get_feature_importance


SIMULATION = False # quick run for testing purposes
OUTPUT_PATH = "./output/"
DATA_VARIANT = "complete"

max_samples_SHAP = 99

#def main():

start = time.time()    
X, y = get_data(DATA_VARIANT)


# create directory and report file
report_dir = OUTPUT_PATH + time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime(start)) + "/"
os.mkdir(report_dir)
report_filename = "report.csv"
with open(report_dir + report_filename , "w", newline="") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';')
    filewriter.writerow(["Type", "Split", "Model", "Train score (R²)", "Test score (R²)","Finished at timestamp (s)"])
    
    
# iterate over regressor types
for reg_type in ["ElasticNet",
                 "LGBMRegressor",
                 "RandomForestRegressor",
                 "GradientBoostingRegressor",
                 "ExtraTreesRegressor",
                 ]:

    reg, hyper_space = get_regressor(reg_type)
    outer_cv = GroupShuffleSplit(n_splits=5 if SIMULATION==False else 2,
                                 test_size=0.2,
                                 random_state=0)    

    # iterate over outer CV splitter
    for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y, groups=X.index), start=1):
    
        y_train, y_test = split_train_test(y, i_train, i_test)
        X_train, X_test = split_train_test(X, i_train, i_test)
    
        # nested CV with parameter optimization
        search_reg = BayesSearchCV(
            estimator=reg,
            search_spaces=hyper_space,
            n_iter=200 if SIMULATION == False else 10,
            cv=5 if SIMULATION == False else 2,
            n_jobs=-2,
            random_state=0
        )
    
        result = search_reg.fit(X_train, y_train.values.ravel())

        # compute feature importance and write to json files
        m_imp,s_imp = get_feature_importance(reg_type, search_reg.best_estimator_, X_test, max_samples_SHAP)           
        
        with open(report_dir + reg_type + str(i_cv) + '_m_imp.json', 'w') as f:
            json.dump(str(m_imp), f)
        with open(report_dir + reg_type + str(i_cv) + '_s_imp.json', 'w') as f:
            json.dump(str(s_imp), f)
            
        # write results to report file
        with open(report_dir + report_filename, "a", newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=';')
            filewriter.writerow(
                (
                    reg_type, 
                    i_cv, 
                    result.best_estimator_, 
                    result.score(X_train, y_train), 
                    result.score(X_test, y_test), 
                    time.time()-start,
                    )
                )
    
    # create hyperparamater optimization plots
    try:
        plot_convergence(search_reg.optimizer_results_)
        plt.savefig(report_dir + reg_type + "_plot_convergence.png")
    except:
        with open(report_dir + reg_type + "_plot_convergence.log", "w") as fo:
            fo.write(traceback.format_exc())
    
    try:
        plot_evaluations(search_reg.optimizer_results_[0])
        plt.savefig(report_dir + reg_type + "_plot_evaluations.png")
    except:
        with open(report_dir + reg_type + "_plot_evaluations.log", "w") as fo:
            fo.write(traceback.format_exc())
    
    try:
        plot_objective(search_reg.optimizer_results_[0])
        plt.savefig(report_dir + reg_type + "_plot_objective.png")
    except:
        with open(report_dir + reg_type + "_plot_objective.log", "w") as fo:
            fo.write(traceback.format_exc())
        

    # # scatter + regression line plots
    # fig, ax = plt.subplots()
    # ax.scatter(X, y, alpha=0.3)
    # x = pd.DataFrame(np.linspace(0, 5), columns=X.columns)
    # ax.plot(x.values, result.best_estimator_.predict(x))
    # ax.title.set_text(reg_type)
    # try:
    #     plt.text(
    #         0.4, 25, f"y = {result.best_estimator_.coef_[0].round(2)}x + {result.best_estimator_.intercept_.round(2)}", fontsize=12)
    # except:
    #     ...
    # plt.text(
    #     0.4, 22, f"Best score $R²$ = {result.best_score_.round(3)} ($r$ = {(result.best_score_.round(3)**(1/2)).round(2)})", fontsize=12)
    # ax.set_ylim(0, 30)
    # ax.set_xlim(0, 5)
    # fig.savefig("predictions_"+reg_type+".png")


print(f"Overall execution time: {(time.time()-start):.3f}s")


# if __name__ == "__main__":
    # main()



