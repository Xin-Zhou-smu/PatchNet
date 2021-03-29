import argparse
import pickle as pickle 
from train import train_model
from evaluation import evaluation_model,testing_model
from preprocessing import reformat_commit_code
from padding import padding_commit, padding_testing_commit
import sys
from sklearn.model_selection import train_test_split
from collections import Counter
from utils import  build_pos_neg_balance

def read_args_PNExtend():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training PatchNet model')   
    parser.add_argument('-valid', action='store_true', help='do validation when training PatchNet model, return the best performing models in validation set')  

    parser.add_argument('-train_data', type=str, default='./m_data/train.pkl', help='the directory of our training data')    
    parser.add_argument('-dictionary_data', type=str, default= './m_data/dict.pkl' , help='the directory of our dicitonary data')
    
    parser.add_argument('-train_valid_ratio', type=float, default= 0.3 , help='the ratio of training set and valid set')


    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str,default= './m_data/testing.pkl' ,help='the directory of our testing data')    

    # Predicting our data
    parser.add_argument('-load_model', type=str,default='./snapshot/2021-01-15_10-48-15/epoch_5.pt', help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('--msg_length', type=int, default=512, help='the length of the commit message')
    parser.add_argument('--code_file', type=int, default=2, help='the number of files in commit code')
    parser.add_argument('--code_hunk', type=int, default=5, help='the number of hunks in each file in commit code')
    parser.add_argument('--code_line', type=int, default=8, help='the number of LOC in each hunk of commit code')
    parser.add_argument('--code_length', type=int, default=32, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('--embedding_dim', type=int, default=8, help='the dimension of embedding vector')
    parser.add_argument('--filter_sizes', type=str, default='1, 2', help='the filter size of convolutional layers')
    parser.add_argument('--num_filters', type=int, default=32, help='the number of filters')
    parser.add_argument('--hidden_units', type=int, default=128, help='the number of nodes in hidden layers')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default = 50, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')

    # Config tensorflow
    parser.add_argument('--allow_soft_placement', type=bool, default=True, help='allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log placement of ops on devices')
    return parser

if __name__ == '__main__':
    params = read_args_PNExtend().parse_args()    
    if params.train is True:
        train_data = pickle.load(open(params.train_data, 'rb'))
        '''
        print(len(train_data))
        print(type(train_data))
        print(train_data[0])
        '''
        #split train & valid set
        train_data, valid_data = train_test_split(train_data, test_size=params.train_valid_ratio, random_state=42)
        

        print('------------------------------------------')
        print('Training set size:', len(train_data))
        print('Validation set size:', len(valid_data))
        print('Train-valid ratio:',params.train_valid_ratio)
        print('------------------------------------------')

        
        print('------------------------This is Training session---------------------------------')
        
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary        

        train_data = reformat_commit_code(commits=train_data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels = padding_commit(commits=train_data, dictionary=dictionary, params=params)          
        
        data = (train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg, dict_code)  
        print("count of label in training set before balancing:", Counter(train_labels))
        
        balanced_data = build_pos_neg_balance(data)
        print("\n # of trainining data after balancing:", len(balanced_data[0]))
        print("count of label in balanced training set:", Counter(balanced_data[3]))
        
        train_model(data = balanced_data, valid_data = valid_data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the training process---------------------------')
        print('--------------------------------------------------------------------------------')
        sys.exit()
        
        
    elif params.predict is True:
        
        print('----------This is testing session-------------')
        pred_data = pickle.load(open(params.pred_data, 'rb'))
        print('-------Test size:', len(pred_data)) 
              
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary       
        
        #print(type(pred_data))
        #print(pred_data[1])
        
        pred_data = reformat_commit_code(commits=pred_data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
        '''
        print(type(pred_data))
        for i in range(10):
            print(pred_data[i])
        '''
        pred_pad_msg, pred_pad_added_code, pred_pad_removed_code, pred_labels, pred_ids = padding_testing_commit(commits=pred_data, dictionary=dictionary, params=params)    
        
        '''
        print(pred_ids[0:2])
        print(pred_ids[0+64:2+64])
        print(pred_ids[0+64*2:2+64*2])
        '''
        
        print(len(pred_ids),len(pred_labels))
        
        data = (pred_pad_msg, pred_pad_added_code, pred_pad_removed_code, pred_labels, dict_msg, dict_code, pred_ids)  
        testing_model(data=data, params=params)
        print('--------------------------------------------------------------------------------')
        print('--------------------------Finish the evaluation process---------------------------')
        print('--------------------------------------------------------------------------------')
        sys.exit()
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        sys.exit()
