import os
import pickle
import warnings
import json
import sys
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
