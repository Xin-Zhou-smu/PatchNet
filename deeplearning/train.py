from utils import mini_batches, save
import os
import datetime
import torch
from model import PatchNet
import torch.nn as nn
from tqdm import tqdm
import pickle as pickle
from preprocessing import reformat_commit_code
from padding import padding_commit
from evaluation import evaluation_model,valid_model

def train_model(data, valid_data, params):
    
    # prepare data for validation
    if params.valid is True:
         print('\n')
        
         
         best_f1 = 0  ## use f1 score to choose the best epoch
         best_model_name = None
         
         valid_dictionary = pickle.load(open(params.dictionary_data, 'rb'))
         val_dict_msg, val_dict_code = valid_dictionary       

         valid_data = reformat_commit_code(commits=valid_data, num_file=params.code_file, num_hunk=params.code_hunk, 
                                num_loc=params.code_line, num_leng=params.code_length)
         valid_pad_msg, valid_pad_added_code, valid_pad_removed_code, valid_labels = padding_commit(commits=valid_data, dictionary=valid_dictionary, params=params)           
        
         valid_data = (valid_pad_msg, valid_pad_added_code, valid_pad_removed_code, valid_labels, val_dict_msg, val_dict_code)
         
    
    
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, 
                                        Y=labels, mini_batch_size=params.batch_size, shuffled=True)

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #params.save_dir = os.path.join(params.save_dir,'2021-01-15_12-13-55')
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNet(args=params)    

    if torch.cuda.is_available():
        model = model.cuda()
      
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()
    
    for epoch in range(1, params.num_epochs + 1):
        
        total_loss = 0
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_added_code, pad_removed_code, labels = batch
            
            
            
            if torch.cuda.is_available():                
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:            
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_added_code).long(), torch.tensor(pad_removed_code).long(),torch.tensor(labels).float()
            
            
            optimizer.zero_grad()
            
            predict = model.forward(pad_msg, pad_added_code, pad_removed_code)
            loss = criterion(predict, labels)            
            loss.backward()
            total_loss += loss
            optimizer.step()            
                        
        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))            
        save(model, params.save_dir, 'epoch', epoch)  
        
    
    # Validation on All models
    #if params.valid is True and (epoch ==params.num_epochs) :
    if params.valid is True :
            print ('\n')
            print('-------------------this is validation session:-------------------------')
            model_paths = os.listdir(params.save_dir)
            model_paths.sort(key = lambda x :int(x[6:-3]))
            print('these are all model files:', model_paths)
            best_model_paths = os.path.join(params.save_dir, 'best_models')
            for model_ in model_paths:
                model_path = os.path.join(params.save_dir, model_)
                #print('model path:', model_path)
                accuracy, roc_score, prc, rc, f1 = valid_model(data= valid_data, params=params, model_path = model_path)
                print('------------------validation result for',model_path)
                print('validation data -- Accuracy: %.4f -- AUC: %.4f -- Precision: %.4f -- Recall: %.4f -- F1: %.4f' % (accuracy, roc_score, prc, rc, f1))
                print('\n')
            
            
                ## select best models in these epoch
                if best_f1 < f1:
                    best_f1 = f1
                    best_model_name = model_
                    print("best models at right now is:", model_)
                    print("\n")
            print('-------best model at last is:', best_model_name)
            print('-----------Notice---------------')
            print('please choose last two or three best model to load when using -load_model option.')
            print('best models are only the recommendation based on their performance on validation set, may not be the real best models')
                
