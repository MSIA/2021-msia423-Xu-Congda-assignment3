import logging

import sklearn
import sklearn.ensemble

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, initial_features, n_estimators, max_depth):
    """train the random forest model
        Args:
            X_train: X varaibles of train dataset
            y_train: y variable of train dataset
            initial_features: features selected to be trained
            n_estimators: number of trees in the forest
            max_depth: maximum depth of trees
        Returns:
            rf: the trained random forest model
    """
    logger.info("Training the model")
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train[initial_features], y_train)
    return rf