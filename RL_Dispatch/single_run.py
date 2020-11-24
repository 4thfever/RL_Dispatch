import yaml

from packages.dqn_learn import dqn_learing
from packages.lib.generator.case_generator import case_generate

with open('config.yaml') as file:
    d = yaml.load(file)
    
def main():
    dqn_learing(d, output_path="666", save_or_plot="save")

if __name__ == '__main__':
    # case_generate(d)
    main()
