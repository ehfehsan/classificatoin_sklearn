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

def type_var(value):
    if value == "np.float":
        return np.float
    elif value == 'object':
        return object

def load_original_data(path="data/exercise_04_train.csv"):
    """
    This function loads the data from the csv file.
    path: path to the csv file.
    df: dataframe object from pandas.
    """
    try:
        df = pd.read_csv(path)
    except IOError:
        print("File does not exist!")

    assert not df.empty, "The csv file is empty."
    return df


class Transformations:
    def __init__(self, path):
        self.path = path
        self.transformations = self.load_transformations_json()
        self.transformations = self.preprocess()

    def load_transformations_json(self):
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
        for _key in self.transformations.keys():
            if not "regex" in self.transformations[_key].keys():
                self.transformations[_key]["regex"] = "False"
            if not "type" in self.transformations[_key].keys():
                self.transformations[_key]["type"] = "None"
            if not "value" in self.transformations[_key].keys():
                print("_key" + "does not have value for transformations!")
                return None
        return self.transformations

    def get_params(self):
        return self.path, self.transformations

    def get_tranformations(self):
        return self.transformations


def create_dataframe_report(data, fn):
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(fn + ".html")


def split_data(data, ratio):
    """
    This function split the data into two subsets based on the passer ratio.
    data: dataframe object from pandas.
    ratio: proportion for splitting the data.
    output: training set instances, training set labels, test set instances, test set labels.
    """
    np.random.seed(13)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    set_1_indices = shuffled_indices[test_set_size:]
    set_2_indices = shuffled_indices[:test_set_size]

    return data.iloc[set_1_indices], data.iloc[set_2_indices]


class SelectFeatureType(BaseEstimator, TransformerMixin):
    """
    This class defined to filter columns from a dataframe based on the type of the column.
    """

    def __init__(self, feature_type):
        self._feature_type = feature_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.select_dtypes(include=[self._feature_type])


class CategoricalValueCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, path):
        self.path = path
        self.transformations = Transformations(path=path)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
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
                df[_key] = df[_key].astype(type_var(value=_value["type"]))
        return x


def calculate_auc(model, x_, y_):
    """
    This function calculates the auc score of the model.
    """
    y_pred = model.predict(x_)
    auc_score = roc_auc_score(y_, y_pred)
    return auc_score


def creat_full_pipeline():
    """
    This function includes all the prepossessing steps for both categorical and numerical features.
    """
    categorical_pipeline_1 = Pipeline(steps=[('cat_selector', SelectFeatureType('object')),
                                             ('cat_transformer', CategoricalValueCorrector(path = "params/replace.json"))
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

    full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                   ('numerical_pipeline', numerical_pipeline)])
    return full_pipeline


def create_train_test_valid(path="data/exercise_04_train.csv"):
    """
    This function creates train, test and validation set and operate preprocessing pipeline on the data.
    """
    df = load_original_data(path)
    data_train, data_test = split_data(data=df, ratio=0.3)
    x_train, y_train = data_train.iloc[:, :-1], data_train.iloc[:, -1]
    x_test, y_test = data_test.iloc[:, :-1], data_test.iloc[:, -1]

    data_train, data_valid = split_data(data=data_train, ratio=0.2)
    x_train, y_train = data_train.iloc[:, :-1], data_train.iloc[:, -1]
    x_valid, y_valid = data_valid.iloc[:, :-1], data_valid.iloc[:, -1]

    f = creat_full_pipeline()
    x_train = f.fit_transform(x_train)
    x_valid = f.fit_transform(x_valid)
    x_test = f.fit_transform(x_test)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def calculate_probability(model, x_, fn):
    """
    This function calculates the prediction probability for the Ridge model.
    """
    prediction_prob = model.predict(x_)
    prob = np.exp(prediction_prob) / (1 + np.exp(prediction_prob))
    np.savetxt(fn, prob)
    return prob


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
                                             ('cat_transformer', CategoricalValueCorrector(path = "params/replace.json"))
                                             ])

    print(df.select_dtypes(include=['object']).head(5))
    categorical_pipeline_1.fit_transform(df)
    print(df.select_dtypes(include=['object']).head(5))

    # train_ridge()
    # train_svm()
