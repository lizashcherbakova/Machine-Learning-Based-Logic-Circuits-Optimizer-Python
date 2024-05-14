from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from plain_model.plain_model import build_and_train_model, build_and_train_model2, build_and_train_model3
from plain_model.prepare_data import split_data_by_scheme_and_design_ready
from utility.data_loader import load_data


# Define your models and parameter sets in dictionaries
models = {
    'RandomForest': build_and_train_model,
    'XGBRegressor': build_and_train_model2,
    'PolynomialFeatures': build_and_train_model3
}

data_splitters = {
    'PlainParameters': split_data_by_scheme_and_design_ready,
    'NPNParameters': split_data_by_scheme_and_design_ready
}


def cross_validate(schemes, designs, scheme_info, models, data_splitters, n_splits=3):
    kf_designs = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf_schemes = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for model_name, model_func in models.items():
        for splitter_name, splitter_func in data_splitters.items():
            for design_fold_idx, (train_index_design, test_index_design) in enumerate(kf_designs.split(designs)):
                train_designs = [designs[i] for i in train_index_design]
                test_designs = [designs[i] for i in test_index_design]
                for scheme_fold_idx, (train_index_scheme, test_index_scheme) in enumerate(kf_schemes.split(schemes)):
                    train_schemes = [schemes[i] for i in train_index_scheme]
                    test_schemes = [schemes[i] for i in test_index_scheme]

                    # Загрузка данных, обучение модели и расчет MSE
                    X_train, X_test, y_train, y_test, _, _ = splitter_func(
                        train_designs, train_schemes, test_designs, test_schemes, scheme_info
                    )
                    model = model_func(X_train, X_test, y_train, y_test)

                    # Предсказание и вычисление MSE для текущего фолда
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)

                    # Save results along with design and scheme information
                    results.append({
                        'Model': model_name,
                        'Splitter': splitter_name,
                        'DesignFold': design_fold_idx,
                        'SchemeFold': scheme_fold_idx,
                        'MSE': mse,
                        'TrainDesigns': train_designs,
                        'TestDesigns': test_designs,
                        'TrainSchemes': train_schemes,
                        'TestSchemes': test_schemes,
                    })

    return results


def save_results_to_csv(results, filename="mse_results.csv"):
    data = []
    for result in results:
        data.append([
            result['Model'], result['Splitter'], result['DesignFold'], result['SchemeFold'], result['MSE'],
            len(result['TrainDesigns']), len(result['TestDesigns']),
            len(result['TrainSchemes']), len(result['TestSchemes']),
            ','.join(result['TrainDesigns']), ','.join(result['TestDesigns']),
            ','.join(result['TrainSchemes']), ','.join(result['TestSchemes'])
        ])

    df = pd.DataFrame(data, columns=[
        "Model", "Splitter", "DesignFold", "SchemeFold", "MSE",
        "NumTrainDesigns", "NumTestDesigns", "NumTrainSchemes", "NumTestSchemes",
        "TrainDesigns", "TestDesigns", "TrainSchemes", "TestSchemes"
    ])
    df.to_csv(filename, index=False)


def main():
    # Load data
    schemes = load_data("schemes")
    designs = load_data("designs")
    scheme_info = load_data("scheme_info")

    # Cross-validate models
    results = cross_validate(schemes, designs, scheme_info, models, data_splitters)

    # Save results to CSV
    save_results_to_csv(results)


# Mean Squared Error: 42880665.31203778
# Mean Squared Error: 254862839.27437794
# Mean Squared Error: 28552203.15020233
# Mean Squared Error: 174164747.6196954
# Mean Squared Error: 88747002.42690513
# Mean Squared Error: 38553851.04273862
if __name__ == "__main__":
    main()
