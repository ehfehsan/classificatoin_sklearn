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
