import pickle

# Path to the .pickle file
file_path = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/OPENABC2_DATASET-2/origAIGNodesDict.pickle'
file_path = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/OPENABC2_DATASET-2/synthesisStatistics.pickle'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the contents of the data
print(data)