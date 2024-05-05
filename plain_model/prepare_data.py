from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from model_prepare.retrive_parameters import get_simple_parameters, get_steps_scheme_optimization, \
    get_simple_area_after_optimization


def prepare_data(schemes, designs) -> Tuple[np.ndarray, np.ndarray, list]:
    features = []
    targets = []
    details = []

    for design in designs:
        for scheme in schemes:
            initial_features = get_simple_parameters(scheme)
            steps = get_steps_scheme_optimization(design)

            input_features = initial_features + steps
            features.append(input_features)

            targets.append(get_simple_area_after_optimization(scheme, design))
            details.append((scheme, design))

    return np.array(features), np.array(targets), details


def split_prepared_data(features, targets, details):
    X_train, X_test, y_train, y_test, details_train, details_test = train_test_split(
        features, targets, details, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, details_train, details_test