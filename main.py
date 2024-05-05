import torch
import torch_geometric
from torch_geometric.data import Data

print("torch version: ", torch.__version__)
print(torch_geometric.__version__)

path = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt/ac97_ctrl/ac97_ctrl_syn0_step20.pt'
path_1 = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/OPENABC2_DATASET-2/processed/ac97_ctrl_syn1_step0.pt'
path_2 = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/OPENABC2_DATASET-2/processed/ac97_ctrl_syn2_step0.pt'
path_3 = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt/fir/fir_syn2_step20.pt'
# data = torch.load(path)
# print(data)

def print_pt(path):
    def bump(g):
        return Data.from_dict(g.__dict__)

    old_data = torch.load(path)
    new_data = bump(old_data)
    print(new_data)
    a = new_data.synVec
    print(new_data.synVec)

print('\nONLY 20th step')
print_pt(path)
print_pt(path_3)

print()

print("ALL steps or only one?")
print_pt(path_1)
print_pt(path_2)

# In PyG 2.*
# data_dict = torch.load(path)
# data = Data.from_dict(data_dict)
# print(data)
#
# # In PyG 1.*
# data = torch.load(path)
# torch.save(data.to_dict(), 'data_dict.pt')