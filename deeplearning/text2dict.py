# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:25:28 2020

@author: xinzhou.2020
"""

## Note that you need to change the path_data to the path of "*.out"
## Input:  path of "*.out" 
## Output: path of "*.pkl"

import argparse
from preprocessing import extract_commit
import pickle

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-text_path', type=str, help='the directory of text data')
    parser.add_argument('-dict_path', type=str, help='the path the generated data want to save')
    parser.add_argument('-print', type=bool,default = False, help='whether to print some example of generated data')

    return parser


if __name__ == "__main__": 
    
    params = read_args().parse_args() 
    
    #path_input_data = "test_data.out"
    #path_output_data = "./try_data/test_try.pkl"
    path_input_data = params.text_path
    path_output_data = params.dict_path
    show_data = params.print                # print the generate data or not
    
    #read commits and change format into dict
    commits_ = extract_commit(path_file=path_input_data)
    #print(commits_[0])
    
    # store dict data
    with open(path_output_data, 'wb') as handle:
        pickle.dump(commits_, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    # print out the generated data
    if (show_data == True):
        print("The first two commits are shown below: ")
        f = open(path_output_data,'rb')
        data = pickle.load(f)
        print(data[0:2])
