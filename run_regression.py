from sklearn.model_selection import ShuffleSplit
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
import codecs, json 
from feature_importance import get_feature_importance
        

SIMULATION = False # quick run for testing purposes
OUTPUT_PATH = "./output/"
DATA_VARIANT = "complete"
N_OUTER_SPLITS = 50
N_INNER_SPLITS = 50

max_samples_SHAP = 99


def main():

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
                     "ExtraTreesRegressor",
                     ]:
    
        reg, hyper_space = get_regressor(reg_type)
        outer_cv = ShuffleSplit(n_splits=N_OUTER_SPLITS if SIMULATION==False else 2,
                                     test_size=0.2)    
    
        # iterate over outer CV splitter
        for i_cv, (i_train, i_test) in enumerate(outer_cv.split(X, y), start=1):
        
            y_train, y_test = split_train_test(y, i_train, i_test)
            X_train, X_test = split_train_test(X, i_train, i_test)
        
            # nested CV with parameter optimization
            inner_cv = ShuffleSplit(n_splits=N_INNER_SPLITS if SIMULATION == False else 2,
                                         test_size=0.2)   
            search_reg = BayesSearchCV(
                estimator=reg,
                search_spaces=hyper_space,
                n_iter=200 if SIMULATION == False else 10,
                cv=inner_cv,
                n_jobs=-2,
                random_state=0
            )
        
            result = search_reg.fit(X_train, y_train.values.ravel())
            
            print(f"Split {i_cv}:", result.best_estimator_)
            print("train score:", round(result.score(X_train, y_train), 5))
            print("test  score:", round(result.score(X_test, y_test), 5))
            print("\n")
    
            # compute feature importance and write to json files
            m_imp,s_imp = get_feature_importance(reg_type, search_reg.best_estimator_, X_test, max_samples_SHAP)       
            json.dump(m_imp.tolist(), codecs.open(report_dir + reg_type + "_" + str(i_cv) +"_m_imp.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) 
            json.dump(s_imp.tolist(), codecs.open(report_dir + reg_type + "_" + str(i_cv) +"_s_imp.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
            
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
            
   
    
    print(f"Overall execution time: {(time.time()-start):.3f}s")


if __name__ == "__main__":
    main()



