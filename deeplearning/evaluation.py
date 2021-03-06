from utils import mini_batches
import torch
from model import PatchNet
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, roc_auc_score
import numpy as np 
import os

def best_accuracy(true_label, pred_proba):
    
    fpr, tpr, thresholds = roc_curve(true_label, pred_proba)
    precision, recall, thresholds = precision_recall_curve(true_label, pred_proba)
    
    num_pos_class = len([1 for l in true_label if l == 1])
    num_neg_class = len([0 for l in true_label if l == 0])
    
    tp = recall * num_pos_class
    fp = (tp / precision) - tp
    tn = num_neg_class - fp
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold

def running_evaluation(model, data):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():        
        predicts, groundtruth = list(), list()
        for i, (batch) in enumerate(tqdm(data)):   
            pad_msg, pad_added_code, pad_removed_code, labels = batch  
            
            #pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)  
            if torch.cuda.is_available():                
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)
            else:            
                pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_added_code).long(), torch.tensor(pad_removed_code).long(),torch.tensor(labels).float()
            
            predicts.append(model.forward(pad_msg, pad_added_code, pad_removed_code))
            groundtruth.append(labels)
            
        predicts = torch.cat(predicts).cpu().detach().numpy()        
        groundtruth = torch.cat(groundtruth).cpu().detach().numpy()
        
        accuracy, _ = best_accuracy(groundtruth, predicts)
        binary_pred = [1 if p >= 0.5 else 0 for p in predicts]  # threshold can be changed 
        
        prc = precision_score(y_true=groundtruth, y_pred=binary_pred)        
        rc = recall_score(y_true=groundtruth, y_pred=binary_pred)
        f1 = f1_score(y_true=groundtruth, y_pred=binary_pred)
        
        #deal with exception case
        if (list(groundtruth).count(1) == len(groundtruth)):
            auc_score = -1
            print('\n')
            print('----------------------------------------------------------------------------')
            print('Cannot compute AUC score because we only have positive samples in test set !')
            print('----------------------------------------------------------------------------')
        elif (list(groundtruth).count(0) == len(groundtruth)):
            auc_score = -1
            print('\n')
            print('---------------------------------------------------------------------------')
            print('Cannot compute AUC score because we only have negative samples in test set !')
            print('---------------------------------------------------------------------------')
        else:
            auc_score = roc_auc_score(groundtruth, predicts)
           
        return accuracy, auc_score, prc, rc, f1


def evaluation_model(data, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, 
                Y=labels, mini_batch_size=params.batch_size, shuffled=False)     

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNet(args=params)    
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()

    accuracy, roc_score, prc, rc, f1 = running_evaluation(model=model, data=batches)    
    print('Test data -- Accuracy: %.4f -- AUC: %.4f -- Precision: %.4f -- Recall: %.4f -- F1: %.4f' % (accuracy, roc_score, prc, rc, f1))    
    


def valid_model(data, params, model_path):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code = data
    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, 
                Y=labels, mini_batch_size=params.batch_size, shuffled=False)     
    '''
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    
    '''
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNet(args=params)    
    
    # use path to load model
    model.load_state_dict(torch.load(model_path))
    #model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()

    accuracy, roc_score, prc, rc, f1 = running_evaluation(model=model, data=batches)    
    #print('Test data -- Accuracy: %.4f -- AUC: %.4f -- Precision: %.4f -- Recall: %.4f -- F1: %.4f' % (accuracy, roc_score, prc, rc, f1))    
    
    return accuracy, roc_score, prc, rc, f1




def testing_model(data, params):
    pad_msg, pad_added_code, pad_removed_code, labels, dict_msg, dict_code, ids = data
    ids = np.array(ids)
    batches = mini_batches(X_msg=pad_msg, X_added_code=pad_added_code, X_removed_code=pad_removed_code, 
                Y=ids, mini_batch_size=params.batch_size, shuffled=False)     
    
    #print(ids[0:10])

    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchNet(args=params)    
    model.load_state_dict(torch.load(params.load_model))
    if torch.cuda.is_available():
        model = model.cuda()
    
    data = batches
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():        
        predicts, id_list = list(), list()
        for i, (batch) in enumerate(tqdm(data)):   
            pad_msg, pad_added_code, pad_removed_code, ids = batch  
            
            #pad_msg, pad_added_code, pad_removed_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda(), torch.cuda.FloatTensor(labels)  
            if torch.cuda.is_available():                
                pad_msg, pad_added_code, pad_removed_code = torch.tensor(pad_msg).cuda(), torch.tensor(pad_added_code).cuda(), torch.tensor(pad_removed_code).cuda()
            else:            
                pad_msg, pad_added_code, pad_removed_code = torch.tensor(pad_msg).long(), torch.tensor(pad_added_code).long(), torch.tensor(pad_removed_code).long()
            
            predicts.append(model.forward(pad_msg, pad_added_code, pad_removed_code))
            id_list.append(ids)
            
        predicts = torch.cat(predicts).cpu().detach().numpy()
        # print(len(predicts), len(id_list))
        
        
        id_list = [ list(id_b) for id_b in id_list]
        patch_ids = []
        for ii in range(len(id_list)):
               patch_ids = patch_ids + id_list[ii]
        #print(len(predicts), len(patch_ids))
        
        result_txt_path = './results/result.txt'
        result_dir = 'results' 
        if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        # check whether exist result.txt 
        if os.path.exists(result_txt_path):
            os.remove(result_txt_path)
        
 
        # 以写的方式打开文件，如果文件不存在，就会自动创建
        file_write_obj = open(result_txt_path, 'w')
        for i in range(len(predicts)):
            patch_name = patch_ids[i]
            scores = predicts[i]
            
            var = str(patch_name) + ' ' + str(scores) + '\n'
            file_write_obj.writelines(var)
        file_write_obj.close()      
