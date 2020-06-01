import os
import pickle
import warnings
import json
import sys
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

# import plotly.express as px

warnings.filterwarnings(action='ignore')


def string_to_vartype(value):
    # Converting string to variable type

    """
    This function returns variable type based on the input string. For example, "np.float" converts to
    np.float.

    :param value: Each column in a dataframe has a type, this is the string name of the variable's variable.
    :type value: string
    :return: variable type based on the input string.
    :rtype: the name of class variable type.

    .. todo:: Todo: Check for a better way to convert string to variable type.

    """

    if value == "np.float":
        return np.float
    elif value == 'object':
        return object


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


class Transformations:
    """Extracting column transformation from json file.
    This class defined for extracting column transformation from a json file. After finalizing
    all necessary transformations for pre-processing of data, a dictionary created and stored in a
    json file. This json file is the input of this class.
    The json file has this format
    {
      column name: {
        "type": "object",
        "regex": True,
        "value": {
          "value1": "converted value1",
          ...
        }
      }
    }
    The "type" key has the return type of column. If the column is "object" type and needed to convert
    to "np.float", then "type": "np.float". The default value is "Default". If the key does not exit,
    it considers as the "Default" value which does not change the column type.
    The "regex" value is for passing regex expression. The default value is "False". If the key does
    not exist, it considers "False".
    The dictionary should have "value" key, otherwise it returns back the error. It inlcudes all
    replace values, for example, "Turday": "Thursday".

    """

    def __init__(self, path):
        """Create a sample of the class.
        Create a sample of the class. It reads the transformation (dictionary) from the json
        file and then pre-process is for missing keys or other issues.
        :param path: path to the json file which includes a dictionary of transformation.
        :type path: str
        :return : class object from type Transformations.
        :rtype : Transformations.
        """
        self.path = path
        self.transformations = self.load_transformations_json()
        self.transformations = self.preprocess()

    def load_transformations_json(self):
        """Loading dictionary from json file.
        This function reads json file which includes all necessary transformation for columns in
        dictionary format.

        :param self: Transformation object sample.
        :type self: Transformations.
        :return transformations: a dictionary of transformations.
        :rtype transformations: dictionary.
        :raises IOError: the file open/read error.
        """
        try:
            if os.path.getsize(self.path) == 0:
                sys.exit(self.path + ' is empty')
            else:
                f = open(self.path, 'r')
        except IOError:
            sys.exit("Error: open/read file: ", self.path)
        transformations = json.load(f)

        return transformations

    def preprocess(self):
        """Pre-processing keys and values from the dictionary of column transformation.
        This function pre-processes loaded dictionary from the json file. If "value" key does not
        exist, it returns back None. If "regex" or "type" value does not exist, default key/value
        is added, "False" and "Default" respectively.
        :param self: Transformation object sample.
        :type self: Transformations.
        :return transformations: processed transformations for columns.
        "rtype transformations: python dictionary.
        """
        transformations = self.transformations
        for _key in transformations.keys():
            if not "regex" in transformations[_key].keys():
                transformations[_key]["regex"] = "False"
            if not "type" in transformations[_key].keys():
                transformations[_key]["type"] = "None"
            if not "value" in transformations[_key].keys():
                print("_key" + "does not have value for transformations!")
                return None
        return transformations

    def get_params(self):
        """Access to Transformations variables.
        This function returns back path and transformations from the Transformations class.
        :param self: Transformation object sample.
        :type self: Transformations.
        :return path: path to the json file.
        :rtype path: str.
        :return transformations: columns transformations.
        :rtype transformations: Python dictionary.
        """
        return self.path, self.transformations

    def get_transformations(self):
        return self.transformations


def create_dataframe_report(data, fn):
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(fn + ".html")


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


class SelectFeatureType(BaseEstimator, TransformerMixin):
    """Selecting columns from datafram based on the type.

    This class defined to filter columns from a dataframe based on the type of the column. This class
    may be used as one step of the pipeline for preprocessing the data.
    :param feature_type: the column type like np.float or object.
    :type feature_type: pandas dataframe column type.
    :return : filtered columns based on the datatype.
    :type : pandas dataframe.
    """

    def __init__(self, feature_type):
        """
        This function creates a sample of the class.
        :param feature_type:  the column type like np.float or object.
        :type feature_type: pandas dataframe column type.
        :return : a class sample from type SelectFeatureType.
        :rtype : SelectFeatureType.
        """
        self._feature_type = feature_type

    def fit(self, x):
        return self

    def transform(self, x):
        """

        :param x: The column of X is filtered based on the input. Itmay be train, test or validation
        set.
        :type x: Pandas dataframe.
        :return : Filtered dataframe based on the feature_type.
        :rtype : Pandas dataframe.
        """
        return x.select_dtypes(include=[self._feature_type])


class CategoricalValueCorrector(BaseEstimator, TransformerMixin):
    """Applying categorical transformation over the dataframe.

    This class defined for preprocessing categorical features of a panda dataframe. This class
    gets transformations in dictionary format and apply them for categorical features. It supports
    both regex and non-regex formats. It also change the type of column in the dataframe based on
    the passed type for each categorical feature. This class may be used as one step in sklearn
    pipeline for preprocessing the data.
    """

    def __init__(self, path):
        """Class initializer

        Creates a sample of CategoricalValueCorrector class.
        :param path: path to the json file which includes dictionary of transformations.
        :type path: str.
        :return : sample of the class CategoricalValueCorrector.
        """
        self.path = path
        self.transformations = Transformations(path=path)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        """Applying transformation over categorical columns of the dataframe.

        This function transforms columns by applying transformations loaded from a json file.
        The transformations is a dictionary object. This function can apply transformations for
        feature values (x) or labels (y). This function supports two types transformations,
        replace value1 with value2 or regex expressions.
        :param x: columns from a dataframe which filtered based on a type.
        :type x: pandas dataframe.
        :param y: label columns from a data frame.
        :type y: pandas dataframe.
        :return x: processed dataframe based on transformations.
        :rtype x: pandas dataframe.
        """
        transformations = self.transformations.get_tranformations()

        for _key, _value in transformations.items():
            if _value["type"] == df[_key].dtype.name or _value["type"] == 'None':
                _value["type"] = "Default"
            if _value["regex"] == "False":
                df.loc[:, _key] = df.loc[:, _key].replace(
                    to_replace=list(_value["value"].keys()),
                    value=list(_value["value"].values()))
            elif _value["regex"] == "True":
                df.loc[:, _key] = df.loc[:, _key].replace(_value["value"].keys(),
                                                          _value["value"].values(),
                                                          regex=True)
            if _value["type"] != "Default":
                df[_key] = df[_key].astype(string_to_vartype(value=_value["type"]))
        return x


def calculate_auc(model, x_, y_):
    """Calculating auc score.
    This function calculates the AUC score based on the passed model for predictions
    and actual labels.
    :param model: Trained model.
    :param x_: predicted values.
    :type x_: dataframe without labels.
    :param y_: actual labels for instances.
    :type y_: pandas dataframe or an array.
    :return auc_score: auc score of the model.
    :rtype auc_score: float.
    """
    y_pred = model.predict(x_)
    auc_score = roc_auc_score(y_, y_pred)
    return auc_score


def creat_full_pipeline():
    """Creating pipeline for transformations.

    This function includes all the prepossessing steps for both categorical and numerical features.
    :return full_pipeline: the pipeline which includes all transformations for categorical and
    numerical features.
    :rtype full_pipeline: sklearn.pipeline.Pipeline
    """
    categorical_pipeline_1 = Pipeline(steps=[('cat_selector', SelectFeatureType('object')),
                                             ('cat_transformer', CategoricalValueCorrector(path="params/replace.json"))
                                             ])

    categorical_pipeline_2 = Pipeline(steps=[('cat_selector', SelectFeatureType('object')),
                                             ('impute_cat', SimpleImputer(strategy="most_frequent")),
                                             ('one_hot_encoder', OneHotEncoder(sparse=False))
                                             ])

    numerical_pipeline = Pipeline(steps=[('num_selector', SelectFeatureType('float64')),
                                         ('impute', SimpleImputer(strategy='median')),
                                         ('std_scalar', StandardScaler())])

    categorical_pipeline = Pipeline(steps=[('categorical_pipeline_1', categorical_pipeline_1),
                                           ('categorical_pipeline_2', categorical_pipeline_2)])

    # Combining transformed categorical and numerical columns.
    full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])
    return full_pipeline


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

    f = creat_full_pipeline()
    x_train = f.fit_transform(x_train)
    x_valid = f.fit_transform(x_valid)
    x_test = f.fit_transform(x_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


# def calculate_probability(model, x_, fn):
#     """
#     This function calculates the prediction probability for the Ridge model.
#     """
#     prediction_prob = model.predict(x_)
#     prob = np.exp(prediction_prob) / (1 + np.exp(prediction_prob))
#     np.savetxt(fn, prob)
#     return prob


def train_ridge():
    """
    This function trains the Ridge model, saves the model, outputs the AUC scores for training, testing and validation
    and save the result of the prediction for the test set in result1.csv.
    """

    x_train, x_valid, x_test, y_train, y_valid, y_test = create_train_test_valid()
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)
    x_poly_test = poly_features.fit_transform(x_test)
    x_poly_valid = poly_features.fit_transform(x_valid)

    if os.path.exists("ridge.sav"):
        ridge = pickle.load(open("ridge.sav", 'rb'))
    else:
        ridge = Ridge()
        ridge.fit(x_poly_train, y_train)
        pickle.dump(ridge, open('ridge.sav', 'wb'))

    print("Ridge, Poly, Train AUC score: {}, Validation AUC score: {}, Test AUC score: {}".
          format(calculate_auc(model=ridge, x_=x_poly_train, y_=y_train),
                 calculate_auc(model=ridge, x_=x_poly_valid, y_=y_valid),
                 calculate_auc(model=ridge, x_=x_poly_test, y_=y_test)
                 ))

    x_test = load_original_data(path='data/exercise_04_test.csv')

    f = creat_full_pipeline()
    x_test = f.fit_transform(x_test)

    x_poly_test = poly_features.fit_transform(x_test)

    prob_test = calculate_probability(model=ridge, x_=x_poly_test, fn="data/ridge.csv")


def train_svm():
    """
    This function trains the SVM model, saves the model, outputs the AUC scores for training, testing and validation
    and save the result of the prediction for the test set in result2.csv.
    """
    x_train, x_valid, x_test, y_train, y_valid, y_test = create_train_test_valid()
    if os.path.exists("svm.sav"):
        svm = pickle.load(open("svm.sav", 'rb'))
    else:
        svm = SVC(gamma="auto", C=0.5, probability=True)
        svm.fit(x_train, y_train)
        pickle.dump(svm, open('svm.sav', 'wb'))

    print(svm)
    print("SVM, Train AUC score: {}, Validation AUC score: {}, Test AUC score: {}".
          format(calculate_auc(model=svm, x_=x_train, y_=y_train),
                 calculate_auc(model=svm, x_=x_valid, y_=y_valid),
                 calculate_auc(model=svm, x_=x_test, y_=y_test)
                 ))

    x_unseen = load_original_data(path='data/exercise_04_test.csv')
    f = creat_full_pipeline()
    x_unseen = f.fit_transform(x_unseen)

    prob_test = svm.predict_proba(x_unseen)
    np.savetxt("data/svm.csv", prob_test)


if __name__ == "__main__":
    # df = load_original_data()
    # create_dataframe_report(data=df, fn="statefarm")
    df = load_original_data()
    categorical_pipeline_1 = Pipeline(steps=[('cat_selector', SelectFeatureType('object')),
                                             ('cat_transformer', CategoricalValueCorrector(path="params/replace.json"))
                                             ])

    print(df.select_dtypes(include=['object']).head(5))
    categorical_pipeline_1.fit_transform(df)
    print(df.select_dtypes(include=['object']).head(5))

# train_ridge()
# train_svm()
