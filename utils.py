def split_train_test(data, i_train, i_test):
    train = data.iloc[i_train, :].values
    test = data.iloc[i_test, :].values
    return train, test
