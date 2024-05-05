import pickle


def save_model(model_name, model):
    filename = model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_model(model_name):
    filename = model_name + '.sav'
    with open(filename, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content


def save_data(data, data_name):
    """
    Saves any serializable data to a file.

    Args:
    data (Serializable): The data to be saved.
    data_name (str): The base name for the file without extension.
    """
    filename = f'{data_name}.pkl'  # Change file extension to .pkl to indicate a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


def load_data(data_name):
    """
    Loads data from a pickle file.

    Args:
    data_name (str): The base name for the file from which to load data, without extension.

    Returns:
    Serializable: The data loaded from the file.
    """
    filename = f'{data_name}.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data


def save_batch(batch_name, X_train, X_test, y_train, y_test, details_train, details_test):
    """
    Saves a batch of datasets into a single file.

    Args:
    X_train, X_test, y_train, y_test (np.ndarray): Train and test datasets.
    details_train, details_test (list): Additional details for train and test datasets.
    batch_name (str): The base name for the batch file.
    """
    batch_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'details_train': details_train,
        'details_test': details_test
    }
    save_data(batch_data, batch_name + "batch")
    print(f"Batch data saved under the name {batch_name}.pkl")


def load_batch(batch_name):
    """
    Loads a batch of datasets from a file.

    Args:
    batch_name (str): The base name for the batch file to load.

    Returns:
    dict: A dictionary containing all parts of the batch.
    """
    batch_data = load_data(batch_name + "batch")
    print(f"Batch data loaded from {batch_name}.pkl")
    return batch_data
