import os
import sys
import argparse
import pandapower as pp

def parse_args(argv):
    parser = argparse.ArgumentParser(description='This tool is a grid constructor for pandapower.')
    parser.add_argument('-i', '--input', help='input txt file path', type=str)
    args = parser.parse_args(argv)
    return args

def grid_constructor(input=None):
    input_file = input

    network = pp.create_empty_network()

    with open(input_file, 'r') as inf:
        # flag for read txt file, 0 = origin state, 1 = waiting for parameters list, 2 = waiting for tables
        READ_FLAG = 0
        for line in inf.readlines():
            if line[:2] == '@@':
                part_name = line[2:].strip()
                READ_FLAG = 1
            elif READ_FLAG == 1:
                param_name = line.strip().split(',')
                READ_FLAG = 2
            elif READ_FLAG == 2:
                params = line.strip().split(',')
                params = ['None' if param=='' else param for param in params]
                if len(params) == len(param_name):
                    param_string = ''
                    for i in range(len(params)):
                        param_string += param_name[i] + '=' + params[i] + ','
                    eval('pp.create_'+part_name+'(' + param_string[:-1] + ',net=network)')
                elif len(params)==1:
                    READ_FLAG = 0
                    param_name = list()
    return network

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    grid_constructor(args.input)
