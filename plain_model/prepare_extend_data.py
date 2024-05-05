from typing import List

import numpy as np

from model_prepare.retrive_npn_parameters import get_npn_counter_parameters


def extend_data(features: np.ndarray, details: List[tuple]) -> np.ndarray:
    extended_features = []

    for i, detail in enumerate(details):
        scheme, _ = detail
        npn_features = get_npn_counter_parameters(scheme)
        extended_feature = np.append(features[i], npn_features)
        extended_features.append(extended_feature)

    return np.array(extended_features)