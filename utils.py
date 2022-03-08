def split_train_test(data, i_train, i_test):
    train = data.iloc[i_train, :]
    test = data.iloc[i_test, :]
    return train, test
