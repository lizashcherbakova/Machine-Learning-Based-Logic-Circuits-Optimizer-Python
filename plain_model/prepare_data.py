from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from model_prepare.retrive_parameters import get_simple_parameters, get_steps_scheme_optimization, \
    get_simple_area_after_optimization


def prepare_data(schemes, designs):
    data_by_design = {}
    for design in designs:
        features = []
        targets = []
        details = []
        for scheme in schemes:
            initial_features = get_simple_parameters(scheme)
            steps = get_steps_scheme_optimization(design)

            input_features = initial_features + steps
            features.append(input_features)

            targets.append(get_simple_area_after_optimization(scheme, design))
            details.append((scheme, design))

        data_by_design[design] = (np.array(features), np.array(targets), details)
    return data_by_design


def split_data_by_design(data_by_design, test_size=0.2):
    designs = list(data_by_design.keys())
    train_designs, test_designs = train_test_split(designs, test_size=test_size, random_state=42)

    def collect_data(selected_designs):
        features = []
        targets = []
        details = []
        for design in selected_designs:
            f, t, d = data_by_design[design]
            features.extend(f)
            targets.extend(t)
            details.extend(d)
        return np.array(features), np.array(targets), details

    X_train, y_train, details_train = collect_data(train_designs)
    X_test, y_test, details_test = collect_data(test_designs)

    return X_train, X_test, y_train, y_test, details_train, details_test