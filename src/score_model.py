import logging

import sklearn

logger = logging.getLogger(__name__)

def score_model(rf, X_test, initial_features):
    """make predictions on the test dataset based on the trained random forest model
        Args:
            rf: the trained random forest model
            X_test: X varaibles of test dataset
            initial_features: features selected to be trained
        Returns:
            ypre_proba_test: predicted probability for test dataset
            ypre_bin_test: predicted classification for test dataset
    """
    logger.info("Scoring the model")
    ypred_proba_test = rf.predict_proba(X_test[initial_features])[:, 1]
    ypred_bin_test = rf.predict(X_test[initial_features])
    return ypred_proba_test, ypred_bin_test