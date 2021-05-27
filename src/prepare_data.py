import logging

import numpy as np

logger = logging.getLogger(__name__)

def get_features(data, columns):
    """select features from raw dataframe
        Args:
            data: the dataframe to select features from
            columns: the feature columns to be selected
        Returns:
            features: a dataframe with the selected feature columns
    """
    logger.info("Getting features")
    features = data[columns]
    return features

def additional_features(features, additional):
    """Add additional features to dataframe of selected features
        Args:
            features: the dataframe with selected features
            additional: the additional features to be added to the features dataframe
        Returns:
            features: a dataframe with all selected features
    """
    for feature in additional:
        if feature == 'log_entropy':
            features['log_entropy'] = features.visible_entropy.apply(np.log)
        elif feature == 'entropy_x_contrast':
            features['entropy_x_contrast'] = features.visible_contrast.multiply(features.visible_entropy)
        elif feature == 'IR_range':
            features['IR_range'] = features.IR_max - features.IR_min
        elif feature == 'IR_norm_range':
            features['IR_norm_range'] = (features.IR_max - features.IR_min).divide(features.IR_mean)
        else:
            raise ValueError('This additional feature cannot be added')
    return features

def get_target(data, target_name):
    """select target from raw dataframe
        Args:
            data: the dataframe to select target from
            target_name: the target column to be selected
        Returns:
            target: a series of selected target column
    """
    logger.info("Getting target")
    target = data[target_name]
    return target
