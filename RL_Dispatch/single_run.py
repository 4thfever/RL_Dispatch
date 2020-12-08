import yaml

from packages.dqn_learn import dqn_learing
from packages.lib.generator.generator import case_generate, grid_generate

with open('config.yaml') as file:
    d = yaml.load(file)
    
def main():
    dqn_learing(d, output_path="666", save_or_plot="save")

if __name__ == '__main__':
    # grid_generate("case14", "14nodes")
    # case_generate(d, case='train')
    # case_generate(d, case='test')
    main()
