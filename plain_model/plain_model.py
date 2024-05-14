from sklearn.ensemble import RandomForestRegressor

from plain_model.prepare_data import split_data_by_scheme_dict_and_design
from utility.constants import PLAIN_MODEL_NAME
from utility.data_loader import save_model, save_batch, load_data, save_data

from xgboost import XGBRegressor


def build_and_train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting the test set results
    predictions = model.predict(X_test)

    # Calculating the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model


def build_and_train_model2(X_train, X_test, y_train, y_test):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicting the test set results
    predictions = model.predict(X_test)

    # Calculating the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def build_and_train_model3(X_train, X_test, y_train, y_test, degree=2):
    # Create a pipeline that first transforms data with polynomial features, then fits a linear regression model
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)

    # Predicting the test set results
    predictions = model.predict(X_test)

    # Calculating the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model


if __name__ == "__main__":
    # directory_path_last = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt'
    # schemes, designs = extract_names(directory_path_last)
    # save_data(list(schemes), "schemes")
    # save_data(list(designs), "designs")

    schemes = load_data("schemes")
    designs = load_data("designs")
    print(schemes, designs)

    # scheme_info = load_schemes_dict(designs, schemes)
    # save_data(scheme_info, "scheme_info")
    # exit(0)
    scheme_info = load_data("scheme_info")
    # X_train, X_test, y_train, y_test, details_train, details_test = split_data_by_scheme_and_design(designs, schemes, scheme_info)
    X_train, X_test, y_train, y_test, details_train, details_test = split_data_by_scheme_dict_and_design(designs,
                                                                                                         schemes,
                                                                                                         scheme_info,
                                                                                                         test_size=0.2)
    # 1012500 810000 202500
    print(X_train.shape, X_test.shape)
    save_batch(PLAIN_MODEL_NAME, X_train, X_test, y_train, y_test, details_train, details_test)
    # Mean Squared Error: 106469.6591573827
    # 227417.77584896298
    # 131870.4601622469
    # Mean Squared Error: 115078.54730728394
    # Mean Squared Error: 641200.9398473701
    # Mean Squared Error: 101277.80265502473
    # Mean Squared Error: 652923.5662191233
    # Mean Squared Error: 652923.5662191233
    # Mean Squared Error: 33392465.917393334

    model = build_and_train_model(X_train, X_test, y_train, y_test)
    save_model(PLAIN_MODEL_NAME, model)
