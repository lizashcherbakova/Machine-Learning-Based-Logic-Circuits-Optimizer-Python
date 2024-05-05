import os
import re
from typing import List

import pandas as pd


def init_npn():
    combined_counts = pd.DataFrame()

    for filename in os.listdir(npn_statistics_path):
        if filename.endswith(".csv"):
            match = re.match(r"(.+)_(\d+)\.csv", filename)
            if match:
                scheme_name, node_count = match.groups()
                npn_data = pd.read_csv(os.path.join(npn_statistics_path, filename), sep=';')
                if scheme_name.endswith("_orig"):
                    scheme_name = scheme_name[:-5]
                npn_data['scheme'] = scheme_name
                # concatenated_data.append(df_optimized)
                df = npn_data[['NPN Class', 'Count']].set_index('NPN Class')
                df.rename(columns={'Count': scheme_name}, inplace=True)
                combined_counts = pd.concat([combined_counts, df], axis=1)
    epsilon = 1e-10
    combined_counts.fillna(0, inplace=True)
    return combined_counts


npn_statistics_path = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/npn_statistics/orig'
all_data_npn = init_npn()


def get_npn_counter_parameters(scheme_name: str) -> List[int]:
    return list(all_data_npn[scheme_name])


if __name__ == "__main__":
    scheme_name = "ac97_ctrl"
    design = "syn0"

    npn = get_npn_counter_parameters(scheme_name)
    print("NPN:", npn)

    scheme_name = "aes"
    npn = get_npn_counter_parameters(scheme_name)
    print("NPN:", npn)
