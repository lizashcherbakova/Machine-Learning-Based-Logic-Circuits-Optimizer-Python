import pandas as pd
from sklearn.metrics import mean_squared_error

from utility.constants import PLAIN_MODEL_NAME
from utility.data_loader import load_model


def stat_model_quality(X, y, details, model, model_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse}")

    results_df = pd.DataFrame({
        'Scheme': [d[0] for d in details],
        'Design': [d[1] for d in details],
        'Real Area': y,
        'Predicted Area': predictions
    })
    results_df.to_csv(model_name + '_design_predictions.csv', index=False)


if __name__ == "__main__":
    model = load_model(PLAIN_MODEL_NAME)
    stat_model_quality()
