from plain_model.prepare_extend_data import extend_data
from utility.constants import PLAIN_MODEL_NAME
from utility.data_loader import load_batch

if __name__ == "__main__":
    batch = load_batch(PLAIN_MODEL_NAME)
    X_train = extend_data(batch['X_train'], batch['details_train'])
    X_test = extend_data(batch['X_test'], batch['details_test'])

    test_batch = set([pair[1] for pair in batch['details_test']])
    train_batch = set([pair[1] for pair in batch['details_train']])
    print(len(test_batch))
    print(len(train_batch))
    train_batch.add(test_batch)
    print(len(train_batch))

    print(batch['X_test'].shape, batch['X_test'][:20])
