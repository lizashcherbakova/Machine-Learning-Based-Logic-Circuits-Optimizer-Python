from plain_model import build_and_train_model, build_and_train_model2, build_and_train_model3
from meauser_model import stat_model_quality
from prepare_extend_data import extend_data
from utility.constants import PLAIN_MODEL_NAME, PLAIN_NPN_MODEL_NAME
from utility.data_loader import load_batch, save_model

# Errors are bc names and scripts "plain_model" are the same
if __name__ == "__main__":
    batch = load_batch(PLAIN_MODEL_NAME)
    X_train = extend_data(batch['X_train'], batch['details_train'])
    X_test = extend_data(batch['X_test'], batch['details_test'])

    # 1 012 500  810000  202500
    # 810000    1 636 200
    # (32400, 25) (8100, 25)
    # (32400, 25) (8100, 202)
    print(batch['X_train'].shape, X_test.shape)

    # Mean Squared Error: 101277.80265502473
    simple_model = build_and_train_model(batch['X_train'], batch['X_test'], batch['y_train'], batch['y_test'])
    # Mean Squared Error: 120986.85962539505
    model = build_and_train_model(X_train, X_test, batch['y_train'], batch['y_test'])
    save_model(PLAIN_NPN_MODEL_NAME, model)

    stat_model_quality(batch['X_test'], batch['y_test'], batch['details_test'], simple_model, "simple_model")
    stat_model_quality(X_test, batch['y_test'], batch['details_test'], model, "npn_model")



