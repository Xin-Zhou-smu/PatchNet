import numpy as np
import math
import random
import os 
import torch

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def mini_batches(X_msg, X_added_code, X_removed_code, Y, shuffled=False, mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    if shuffled == True:
        permutation = list(np.random.permutation(m))        
        shuffled_X_msg = X_msg[permutation, :]
        shuffled_X_added = X_added_code[permutation, :, :, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :, :, :]
    else:        
        shuffled_X_msg = X_msg
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code

    if shuffled == True:
        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:        
        shuffled_Y = Y        

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):        
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]        
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:        
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_added = shuffled_X_added[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[num_complete_minibatches * mini_batch_size: m, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]        
        mini_batch = (mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def build_pos_neg_balance(data):

    train_pad_msg, train_pad_added_code, train_pad_removed_code, train_labels, dict_msg, dict_code = data
    train_pad_msg_balanced, train_pad_added_code_balanced, train_pad_removed_code_balanced, train_labels_balanced = list(), list(), list(), list()

    Y_pos = [i for i in range(len(train_labels)) if train_labels[i] == 1]
    Y_neg = [i for i in range(len(train_labels)) if train_labels[i] == 0]

    for pos_index in Y_pos:
        # positive sample
        train_pad_msg_balanced.append(train_pad_msg[pos_index])
        train_pad_added_code_balanced.append(train_pad_added_code[pos_index])
        train_pad_removed_code_balanced.append(train_pad_removed_code[pos_index])
        train_labels_balanced.append(train_labels[pos_index])

        #negative sample
        neg_index = random.sample(Y_neg, 1)[0]
        train_pad_msg_balanced.append(train_pad_msg[neg_index])
        train_pad_added_code_balanced.append(train_pad_added_code[neg_index])
        train_pad_removed_code_balanced.append(train_pad_removed_code[neg_index])
        train_labels_balanced.append(train_labels[neg_index])

    balanced_data = ( np.array(train_pad_msg_balanced),  np.array(train_pad_added_code_balanced),  np.array(train_pad_removed_code_balanced),  np.array(train_labels_balanced), dict_msg, dict_code)

    return balanced_data
