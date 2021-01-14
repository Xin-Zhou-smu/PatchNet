# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:43:29 2020

@author: xinzhou.2020
"""

import argparse
from train import train_model
from evaluation import evaluation_model
from preprocessing import reformat_commit_code, extract_commit
import pickle 
from extracting import extract_msg, extract_code, dictionary

def read_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-text_path1', type=str,default ='training_data.out' ,help='the first directory of text data(training data)')
    parser.add_argument('-text_path2', type=str,default ='test_data.out' ,help='the second directory of text data (test data)')
    parser.add_argument('-dict_path', type=str,default = 'dict_try.pkl' ,help='the path the generated data want to save')
    parser.add_argument('-print', type=bool,default = False, help='whether to print some example of generated data')

    return parser

if __name__ == '__main__':
 
    #path_input_data1 = "test_data.out"
    #path_input_data2 = "training_data.out"
    #path_output_data = "./try_data/dict_try.pkl"
    #show_dict = True
    params = read_args().parse_args() 
    path_input_data1 =  params.text_path1
    path_input_data2 =  params.text_path2
    path_output_data =  params.dict_path
    show_dict = params.print
    
    test_data = extract_commit(path_file=path_input_data1)
    train_data = extract_commit(path_file=path_input_data2)
    
    whole_data = train_data + test_data   # add  train data and test data together
    #whole_data = test_data
    msgs, codes = extract_msg(whole_data), extract_code(whole_data)
    dict_msg, dict_code = dictionary(data=msgs), dictionary(data=codes)
    
    #print(len(msgs))
    #print (msgs[1])
    #print(len(codes))
    #print(codes[1])
    print("the number of different tokens in message part is : {n}".format(n=len(dict_msg)))
    print("the number of different tokens in code part is : {n}".format(n=len(dict_code)))
   

    dict_whole = (dict_msg, dict_code)
    with open(path_output_data, 'wb') as handle:
        pickle.dump(dict_whole, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    if show_dict == True:
        f = open(path_output_data,'rb')
        data = pickle.load(f)
        print(len(data))    # length is 2:  [dict_msg, dict_code]
        print(len(data[0])) # #of different tokens in msg
        print(len(data[1]))
        #print(data[0])     #print dict_msg
        #print(data[1])     #print dict_code
      
