from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from get_inital_data import extract_names
from prepare_data import prepare_data
from utility.constants import PLAIN_MODEL_NAME
from utility.data_loader import save_model, save_batch

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


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
    directory_path_last = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt'
    schemes, designs = extract_names(directory_path_last)
    features, targets, details = prepare_data(schemes, designs)
    X_train, X_test, y_train, y_test, details_train, details_test = train_test_split(
        features, targets, details, test_size=0.2, random_state=42)
    # 1012500 810000 202500
    print(features.size, X_train.size, X_test.size)
    # Mean Squared Error: 106469.6591573827
    #  131870.4601622469
    # Mean Squared Error: 115078.54730728394
    # Mean Squared Error: 641200.9398473701
    # Mean Squared Error: 101277.80265502473

    model = build_and_train_model(X_train, X_test, y_train, y_test)
    save_model(PLAIN_MODEL_NAME, model)
    save_batch(PLAIN_MODEL_NAME, X_train, X_test, y_train, y_test, details_train, details_test)
