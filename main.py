# load dict data

# cycle of 20

# cycle of different models

# split

# train

# collect statistics

from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import KFold

from plain_model.plain_model import build_and_train_model
from plain_model.prepare_data import split_data_by_scheme_and_design, split_data_by_scheme_and_design_ready
from utility.data_loader import load_data


def cross_validate(schemes, designs, scheme_info):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = np.zeros(n_splits)

    fold_idx = 0
    for train_index, test_index in kf.split(designs):
        train_designs = [designs[i] for i in train_index]
        test_designs = [designs[i] for i in test_index]
        for train_index_s, test_index_s in kf.split(schemes):
            train_schemes = [schemes[i] for i in train_index_s]
            test_schemes = [schemes[i] for i in test_index_s]

            # Загрузка данных, обучение модели и расчет MSE
            X_train, X_test, y_train, y_test, _, _ = split_data_by_scheme_and_design_ready(train_designs, train_schemes,
                                                                                           test_designs, test_schemes, scheme_info)
            model = build_and_train_model(X_train, X_test, y_train, y_test)

            # Предсказание и вычисление MSE для текущего фолда
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mse_scores[fold_idx] = mse
            fold_idx += 1

    print("MSE по каждому фолду:", mse_scores)
    print("Среднее MSE:", np.mean(mse_scores))
    print("Стандартное отклонение MSE:", np.std(mse_scores))


def main():
    # Load data
    schemes = load_data("schemes")
    designs = load_data("designs")
    scheme_info = load_data("scheme_info")

    # Cross-validate model
    average_mse = cross_validate(schemes, designs, scheme_info)
    print(average_mse)


if __name__ == "__main__":
    main()
