import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os
import json

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
            sys.exit("Error: open/read file.")
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
            if _value["type"] == x[_key].dtype.name or _value["type"] == 'None':
                _value["type"] = "Default"
            if _value["regex"] == "False":
                x.loc[:, _key] = x.loc[:, _key].replace(
                    to_replace=list(_value["value"].keys()),
                    value=list(_value["value"].values()))
            elif _value["regex"] == "True":
                x.loc[:, _key] = x.loc[:, _key].replace(_value["value"].keys(),
                                                        _value["value"].values(),
                                                        regex=True)
            if _value["type"] != "Default":
                x[_key] = x[_key].astype(string_to_vartype(value=_value["type"]))
        return x
