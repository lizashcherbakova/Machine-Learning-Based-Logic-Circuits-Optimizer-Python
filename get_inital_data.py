import os
import re


def extract_names(directory):
    scheme_names = set()
    synthesis_names = set()
    # Regex to match the pattern 'scheme_synthesis_step.pt'
    pattern = re.compile(r'(.+?)_(syn\d+)_step\d+\.pt')

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            match = pattern.match(filename)
            if match:
                scheme_name, synthesis_id = match.groups()
                scheme_names.add(scheme_name)
                synthesis_names.add(synthesis_id)

    return scheme_names, synthesis_names


if __name__ == "__main__":
    directory_path = '/Users/dreamer1977/ed/work/ispras/vkr/statistics_for_model/openabcd_step20_pt'
    schemes, syntheses = extract_names(directory_path)
    print("Scheme len:", len(schemes))
    print("Scheme Names:", schemes)
    print("Synthesis len:", len(syntheses))
    print("Synthesis Names:", syntheses)
