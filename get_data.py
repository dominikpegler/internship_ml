import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_data():
    """
    loads tab-separated file and returns pandas dataframes for X and y
    """

    df = pd.read_csv("./data/mindfulness.csv", sep='\t', index_col=0)

    X = df[[
        "ffmq_aa1",
        "ffmq_ds2",
        "ffmq_aa2",
        "ffmq_nr2",
        "ffmq_aa3",
        "ffmq_nj1",
        "ffmq_ob1",
        "ffmq_aa4",
        "ffmq_nr3",
        "ffmq_ob2",
        "ffmq_nr4",
        "ffmq_nr5",
        "ffmq_nj2",
        "ffmq_ob3",
        "ffmq_ds3",
        "ffmq_nr6",
        "ffmq_nj3",
        "ffmq_ob4",
        "ffmq_ds4",
        "ffmq_nr7",
        "ffmq_nj4",
        "upps_ur_1",
        "upps_ur_2",
        "upps_ur_3",
        "upps_ur_4",
        "upps_ur_5",
        "upps_pm_1",
        "upps_pm_2",
        "upps_pm_3",
        "upps_pm_4",
        "upps_pm_5",
        "upps_pe_1",
        "upps_pe_2",
        "upps_pe_3",
        "upps_pe_4",
        "upps_pe_5",
        "upps_ss_1",
        "upps_ss_2",
        "upps_ss_3",
        "upps_ss_4",
        "upps_ss_5",
        "dmq_cope_1",
        "dmq_cope_2",
        "dmq_cope_3",
        "dmq_cope_4",
        "dmq_cope_5",
        "geschlecht_kod_male",
        "erwerbstaetig_sub"
    ]]

    y = df[["audit"]]

    return X, y


def get_housing_data():

    df = pd.read_csv("./data/housing.csv")
    y_label = "median_house_value"

    # convert categorial variables to bool
    df = pd.get_dummies(df, prefix="", prefix_sep="")

    # impute missing values
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # rescale the features
    non_numeric_features = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY',
                            'NEAR OCEAN']
    do_not_to_scale = non_numeric_features+[y_label]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(
        df.drop(do_not_to_scale, axis=1)), columns=df.columns.drop(do_not_to_scale))
    df = df_scaled.join(df[do_not_to_scale])

    X = df[df.columns.drop(y_label)]
    y = df[[y_label]]

    return X, y
