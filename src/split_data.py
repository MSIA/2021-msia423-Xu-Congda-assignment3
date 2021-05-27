import logging

import sklearn.model_selection

logger = logging.getLogger(__name__)

def train_test_split(features, target, ratio):
    """split train and test data
        Args:
            features: the dataframe of features
            target: the series of target
            ratio: train-test split ratio
        Returns:
            X_train: X variables of train dataset
            X_test: X variables of test dataset
            y_train: y variable of train dataset
            y_test: y variable of test dataset
    """
    logger.info("Splitting data into train and test")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, test_size=ratio)
    return X_train, X_test, y_train, y_test
