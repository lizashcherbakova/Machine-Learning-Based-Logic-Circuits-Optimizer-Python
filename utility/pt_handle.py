import os
import zipfile

import torch
import torch_geometric
from torch_geometric.data import Data

dict_step_names = {"step0": "/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/OPENABC2_DATASET-2/processed",
                   "step20": "/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt"}


def get_path_name(step_tag: str, scheme_name: str) -> str:
    if step_tag == 'step0':
        return os.path.join(dict_step_names[step_tag])
    else:
        return os.path.join(dict_step_names[step_tag], scheme_name)


def build_scheme_name(step_tag: str, scheme_name: str, design_name: str) -> str:
    return f"{scheme_name}_{design_name}_{step_tag}"


def get_loaded_pt(path):
    def bump(g):
        return Data.from_dict(g.__dict__)

    old_data = torch.load(path)
    new_data = bump(old_data)
    return new_data


def retrieve_pt_dict(scheme_name: str, design_name: str, step_tag: str) -> torch.Tensor:
    # Build the full file path
    file_name = build_scheme_name(step_tag, scheme_name, design_name)
    file_path = os.path.join(get_path_name(step_tag, scheme_name), file_name)
    pt_file_path = file_path + '.pt'

    if not os.path.exists(pt_file_path):
        print(f"File {pt_file_path} not found, checking for zipped version...")
        zip_path = pt_file_path + '.zip'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extracting specific file
            zip_ref.extractall(get_path_name(step_tag, scheme_name))
            print(f"Extracted {file_name} from archive to {get_path_name(step_tag, scheme_name)}")
            if os.path.exists(pt_file_path):
                return get_loaded_pt(pt_file_path)
            else:
                print("File was not found in the zip archive.")
                return None
    else:
        return get_loaded_pt(pt_file_path)


if __name__ == "__main__":
    scheme = "ac97_ctrl"
    design = "syn0"
    step = "step0"
    tensor = retrieve_pt_dict(scheme, design, step)
    if tensor is not None:
        print("Tensor loaded successfully.")
    else:
        print("Tensor could not be loaded.")
