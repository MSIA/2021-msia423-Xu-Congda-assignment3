import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_data(path, columns):
    """load data from a local file and select certain columns
        Args:
            path: the path where the data is to be loaded
            columns: the columns of raw data to be selected
        Returns:
            data: the loaded dataframe
    """
    logger.info("Loading data")
    with open(path, 'r') as f:
        data = [[s for s in line.split(' ') if s != ''] for line in f.readlines()]
    first_cloud = data[53:1077]
    first_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)
    first_cloud['class'] = np.zeros(len(first_cloud))
    second_cloud = data[1082:2105]
    second_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in second_cloud]
    second_cloud = pd.DataFrame(second_cloud, columns=columns)
    second_cloud['class'] = np.ones(len(second_cloud))
    data = pd.concat([first_cloud, second_cloud])
    return data