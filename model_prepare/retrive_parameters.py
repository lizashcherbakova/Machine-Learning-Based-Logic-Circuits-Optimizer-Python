from typing import List

from utility.pt_handle import retrieve_pt_dict


def get_simple_parameters(scheme_name: str) -> list:
    design = "syn0"
    step = "step0"
    data = retrieve_pt_dict(scheme_name, design, step)

    # Extract the initial characteristics you're interested in
    # Here, I'm extracting attributes that might represent "initial characteristics"
    # You need to adjust these according to what's actually available in your data
    characteristics = [
        data.longest_path.item(),  # Assuming these attributes are scalar or can be converted to scalar
        data.and_nodes.item(),
        data.pi.item(),
        data.po.item(),
        data.not_edges.item()
    ]

    return characteristics


def get_steps_scheme_optimization(design_name: str) -> List[int]:
    scheme_name = "ac97_ctrl"
    step = "step0"
    data = retrieve_pt_dict(scheme_name, design_name, step)
    return data.synVec.tolist()


def get_simple_parameters_after_optimization(scheme_name: str, design_name: str) -> List[float]:
    step = "step20"
    data = retrieve_pt_dict(scheme_name, design_name, step)

    # Extract the initial characteristics you're interested in
    # Here, I'm extracting attributes that might represent "initial characteristics"
    # You need to adjust these according to what's actually available in your data
    characteristics = [
        data.longest_path.item(),  # Assuming these attributes are scalar or can be converted to scalar
        data.and_nodes.item(),
        data.pi.item(),
        data.po.item(),
        data.not_edges.item()
    ]

    return characteristics


def get_simple_area_after_optimization(scheme_name: str, design_name: str) -> int:
    step = "step20"
    data = retrieve_pt_dict(scheme_name, design_name, step)
    return data.and_nodes.item()


if __name__ == "__main__":
    scheme_name = "ac97_ctrl"
    design = "syn0"
    initial_characteristics = get_simple_parameters_after_optimization(scheme_name, design)
    print("Initial Characteristics:", initial_characteristics)

    steps = get_steps_scheme_optimization(design)
    print("Steps:", steps, type(steps))
    # 11052
    size = get_simple_area_after_optimization(scheme_name, design)
    print("Result size:", size)
