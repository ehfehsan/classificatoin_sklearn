import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np

def load_original_data(path="data/exercise_04_train.csv"):
    """Converting string to column type in pandas.
    This function loads the raw data from the csv file.

    :param path: string path to the csv file.
    :type path: str
    :return df: loaded data from the csv file.
    :rtype df: Pandas.DataFrame
    :raises IOError: the file exists or not.
    """

    try:
        df = pd.read_csv(path)
    except IOError:
        print("File does not exist!")

    assert not df.empty, "The csv file is empty."
    return df


def split_data(data, ratio):
    """Splitting the data into two parts based on the ratio.

    This function splits the data into two separate files based on the passed ratio.
    :param data: a matrix includes instances and features.
    :type data: pandas dataframe.
    :param ratio: a positive number for splitting the dataset between 0 and 1.
    :type ratio: float.
    :return : sliced data into two parts.
    :rtype : pandas dataframe.
    """
    assert ratio <= 0, "The ratio value should be greater than zero."
    assert ratio > 1, "The ratio value should be less than 1."

    np.random.seed(13)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    set_1_indices = shuffled_indices[test_set_size:]
    set_2_indices = shuffled_indices[:test_set_size]

    return data.iloc[set_1_indices], data.iloc[set_2_indices]


def create_train_test_valid(path="data/exercise_04_train.csv",
                            ratio_train_test = 0.3,
                            ratio_train_valid = 0.2):
    """Creating train, test and validation sets.

    This function creates train, test and validation set and operate preprocessing pipeline
    on them.
    :param path: address to the data file in csv format.
    :type path: str.
    :param ratio_train_test: the ratio for train/test sets. Less than 1, greater than 0.
    :type ratio_train_test: float.
    :param ratio_train_valid: the ratio for train/validation sets. Less than 1, greater than 0.
    :type ratio_train_valid: float.
    :return x_train: Train set.
    :rtype x_train: numpy.array.
    :return x_valid: Validation set.
    :rtype x_valid: numpy.array.
    :return x_test: Test set.
    :rtype x_test: numpy.array.
    :return y_train: Labels for train set.
    :rtype y_train: numpy.array.
    :return y_valid: Labels for validation set.
    :rtype y_valid: numpy.array.
    :return y_test: Labels for test set.
    :rtype y_test: numpy.array.
    """
    data = load_original_data(path)
    data_train_valid, data_test = split_data(data=data, ratio=ratio_train_test)
    x_test, y_test = data_test.iloc[:, :-1], data_test.iloc[:, -1]

    data_train, data_valid = split_data(data=data_train_valid, ratio=ration_train_valid)
    x_train, y_train = data_train.iloc[:, :-1], data_train.iloc[:, -1]
    x_valid, y_valid = data_valid.iloc[:, :-1], data_valid.iloc[:, -1]

    # TODO: this part should be moved outside this function.
    f = creat_full_pipeline()
    x_train = f.fit_transform(x_train)
    x_valid = f.fit_transform(x_valid)
    x_test = f.fit_transform(x_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def create_dataframe_report(data, fn):
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(fn + ".html")
