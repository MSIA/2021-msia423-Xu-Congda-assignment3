import logging

import requests as re

logger = logging.getLogger(__name__)

def acquire_data(url, output_path):
    """Acquire raw data from an url
        Args:
            url: the url where the data is to be acquired
            output_path: the path where the acquired data to be stored
        Returns:
            None
    """
    logger.info("Acquiring data from %s", url)
    cloud = re.get(url)
    open(output_path, 'wb').write(cloud.content)

