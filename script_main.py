import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv("data_export",sep='\t',index_col=0)
    
    X = df.copy()[[
        "ffmq_aa1", "ffmq_ds2", "ffmq_aa2", "ffmq_nr2", "ffmq_aa3",
        "ffmq_nj1", "ffmq_ob1", "ffmq_aa4", "ffmq_nr3", "ffmq_ob2",
        "ffmq_nr4", "ffmq_nr5", "ffmq_nj2", "ffmq_ob3", "ffmq_ds3",
        "ffmq_nr6", "ffmq_nj3", "ffmq_ob4", "ffmq_ds4", "ffmq_nr7",
        "ffmq_nj4", "upps_ur_1", "upps_ur_2", "upps_ur_3", "upps_ur_4",
        "upps_ur_5", "upps_pm_1", "upps_pm_2", "upps_pm_3", "upps_pm_4",
        "upps_pm_5", "upps_pe_1", "upps_pe_2", "upps_pe_3", "upps_pe_4",
        "upps_pe_5", "upps_ss_1", "upps_ss_2", "upps_ss_3", "upps_ss_4",
        "upps_ss_5","dmq_cope_1", "dmq_cope_2", "dmq_cope_3", "dmq_cope_4",
        "dmq_cope_5","geschlecht_kod_male","erwerbstaetig_sub"
    ]]
    
    y = df.copy()["audit"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    #n_samples = X_train.shape[0]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    mod = LinearRegression()
    mod.fit(X_train,y_train)
    
    cross_val_score(mod, X_train, y_train, cv=cv)
    scores = cross_val_score(mod, X_train, y_train,cv=cv)
                            
    print("%0.2f accuracy with a standard deviation of %0.2f"
          % (scores.mean(), scores.std()))

  
if __name__ == "__main__":
    main()