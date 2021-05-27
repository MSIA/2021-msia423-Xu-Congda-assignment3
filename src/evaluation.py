import logging

import sklearn
import pandas as pd

logger = logging.getLogger(__name__)

def evaluation(y_test, ypred_proba_test, ypred_bin_test):
    """evaluate the performance of the random forest model
        Args:
            ypre_proba_test: predicted probability for test dataset
            ypre_bin_test: predicted classification for test dataset
        Returns:
            None
    """
    logger.info('Evaluating performance')
    auc = sklearn.metrics.roc_auc_score(y_test, ypred_proba_test)
    confusion = sklearn.metrics.confusion_matrix(y_test, ypred_bin_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, ypred_bin_test)
    classification_report = sklearn.metrics.classification_report(y_test, ypred_bin_test)
    print('AUC on test: %0.3f' % auc)
    print('Accuracy on test: %0.3f' % accuracy)
    print()
    print(pd.DataFrame(confusion,
                       index=['Actual negative', 'Actual positive'],
                       columns=['Predicted negative', 'Predicted positive']))
    print()
    print(classification_report)