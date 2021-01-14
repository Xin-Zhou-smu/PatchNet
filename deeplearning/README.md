# PatchNet: Hierarchical Deep Learning-Based Stable Patch Identification for the Linux Kernel [[pdf](https://arxiv.org/pdf/1911.03576.pdf)]

## Implementation Environment

Please install the neccessary libraries before running our tool:

- python==3.6.9
- torch==1.2.0
- tqdm==4.46.1
- nltk==3.4.5
- numpy==1.16.5
- scikit-learn==0.22.1

## Data & Pretrained models:

Please following the link below to download the data and pretrained models of our paper. 

- https://drive.google.com/drive/folders/1vO4eF4tma94tsBljLMvVXdG2K4sKOC3s?usp=sharing

After downloading, simply copy the data and model folders to PatchNet folder. 



## Hyperparameters:
We have a number of different parameters

* --embedding_dim: Dimension of embedding vectors.
* --filter_sizes: Sizes of filters used by the convolutional neural network. 
* --num_filters: Number of filters. 
* --hidden_layers: Number of hidden layers. 
* --dropout_keep_prob: Dropout for training. 
* --l2_reg_lambda: Regularization rate. 
* --learning_rate: Learning rate. 
* --batch_size: Batch size. 
* --num_epochs: Number of epochs. 

## Running and evalutation

If you only download our dataset and want to train it by yourself,  please follow the instruction (step1&2) below. 

Or if you want to directly use our pretrained model to reproduce our results, please only use command in the step2 and give the path of our pretrained model to '--load_model'.  
      
Step 1. To train the model for bug fixing patch classification, please follow this command: 

       $ python main.py -train -train_data [path of our data] -dictionary_data [path of our dictionary data]
  For example:
       
       $ python main.py -train -train_data 'train.pkl' -dictionary_data 'dict.pkl'
     
Step 2. To evaluate the model for bug fixing patch classification, please follow this command:
      
       $ python main.py -predict -pred_data [path of our data] -dictionary_data [path of our dictionary data] -load_model [path of our model]
  For example:     
  
       $ python main.py -predict -pred_data 'test.pkl' -dictionary_data 'dict.pkl' -load_model './snapshot/2020-12-01_07-45-03/epoch_20.pt'
  Notes:
    "-load_model"  parameter needs the path to the saved model. In the training phase, PatchNet will automatically save some intermediate models during the training process (when we finish a training process, we can see them), which are stroed in folder "snapshot". In the "snapshot" folder, there are folders named by "year-month-day-hour-minute-second" way, to represent the time when models are stored.
    
     In each "year-month-day-hour-minute-second" folder, there are many intermediate model files are named by "epoch_x.pt" (x is a number). For example, "epoch_20.pt" means the model are saved after training 20 epochs.
     
     We need to load these stored models when doing evaluation. If we load "epocj_20.pt" and do evaluation, that means we only evaluate the performance of the model "epoch_20.pt" (model saved at 20th epoch). Usually we will do evaluation on the model with biggest epoch number (usually the more epoch we trained, the better the trained model is). For example, in PatchNet the largest epoch is 50, so we may load in the model "epoch_50.pt" and evaluate on it.
     

## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg
