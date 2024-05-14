from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split

from model_prepare.retrive_parameters import get_simple_parameters, get_steps_scheme_optimization, \
    get_simple_area_after_optimization


def prepare_data(schemes: List[str], designs: List[str]) -> Dict[
    str, Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]]:
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


def load_schemes_dict(designs, schemes):
    return {
        scheme: {design: get_simple_parameters(scheme) + get_steps_scheme_optimization(design) for design in designs}
        for scheme in schemes}


def split_data_by_scheme_dict_and_design(designs: List[str], schemes: List[str],
                                         schemes_dict: Dict[str, Dict[str, Any]],
                                         test_size: float = 0.2):
    train_designs, test_designs = train_test_split(designs, test_size=test_size, random_state=42)
    train_schemes, test_schemes = train_test_split(schemes, test_size=test_size, random_state=42)

    def collect_data(selected_designs, selected_schemes):
        features = []
        targets = []
        details = []
        for design in selected_designs:
            for scheme in selected_schemes:
                input_features = schemes_dict[scheme][design]

                features.append(input_features)

                targets.append(get_simple_area_after_optimization(scheme, design))
                details.append((scheme, design))

        return np.array(features), np.array(targets), details

    X_train, y_train, details_train = collect_data(train_designs, train_schemes)
    X_test, y_test, details_test = collect_data(test_designs, test_schemes)

    return X_train, X_test, y_train, y_test, details_train, details_test


def split_data_by_scheme_and_design(designs: List[str], schemes: List[str], test_size: float = 0.2):
    train_designs, test_designs = train_test_split(designs, test_size=test_size, random_state=42)
    train_schemes, test_schemes = train_test_split(schemes, test_size=test_size, random_state=42)

    def collect_data(selected_designs, selected_schemes):
        features = []
        targets = []
        details = []
        for design in selected_designs:
            steps = get_steps_scheme_optimization(design)
            for scheme in selected_schemes:
                initial_features = get_simple_parameters(scheme)

                input_features = initial_features + steps
                features.append(input_features)

                targets.append(get_simple_area_after_optimization(scheme, design))
                details.append((scheme, design))

        return np.array(features), np.array(targets), details

    X_train, y_train, details_train = collect_data(train_designs, train_schemes)
    X_test, y_test, details_test = collect_data(test_designs, test_schemes)

    return X_train, X_test, y_train, y_test, details_train, details_test


def split_data_by_scheme_and_design_ready(train_designs, train_schemes: List[str], test_designs,
                                          test_schemes: List[str], schemes_dict: Dict[str, Dict[str, Any]]):
    def collect_data(selected_designs, selected_schemes):
        features = []
        targets = []
        details = []
        for design in selected_designs:
            for scheme in selected_schemes:
                input_features = schemes_dict[scheme][design]

                features.append(input_features)

                targets.append(get_simple_area_after_optimization(scheme, design))
                details.append((scheme, design))

        return np.array(features), np.array(targets), details

    X_train, y_train, details_train = collect_data(train_designs, train_schemes)
    X_test, y_test, details_test = collect_data(test_designs, test_schemes)

    return X_train, X_test, y_train, y_test, details_train, details_test
