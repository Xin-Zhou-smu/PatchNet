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

Or if you want to directly use our pretrained model to reproduce the results, please only use the command in the step2 and give the path of the pretrained model to '--load_model'.  
      
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
     
   We need to load these stored models when doing evaluation. If we load "epoch_20.pt" and do evaluation, that means we only evaluate the performance of the model "epoch_20.pt" (model saved at 20th epoch).
     
## Train on New Dataset
### Collect Data
please refer to the preprocessing folder for details on how to collect dataset
#### Change the format of Dataset
As the collected dataset from the preprocessing folder has a different format, we cannot directly use the data to train and test PatchNet.
It needs to change its format following instructions below:

1.  build dictionary-form dataset from the text-form data     
     
       
      $ python text2dict.py -text_path  [path of text data] -dict_path [path of the dictionary data want to store]  -print True
   
   Examples to generate train.pkl and test.pkl:
      
      $ python text2dict.py -text_path  'train_data.out' -dict_path 'train.pkl'  
      $ python text2dict.py -text_path  'test_data.out' -dict_path 'test.pkl' 
     
     
2. build vocabulury dicntionary from text-form data
      
       $ python generate_dict.py -text_path1 [path of our data1] -text_path2 [path of our data2] -dict_path [path we want to store dict.pkl]
    Examples to generate dict.pkl:
    
       $ python generate_dict.py -text_path1 'training_data.out' -text_path2 'test_data.out' -dict_path 'dict.pkl'
   Notes:
   training_data.out is the "text format" patches as training dataset (used in trainig phase).
   
   test_data.out is the "text format" patches as test dataset (used in evaluation phase).
   
   The reason why we need evaluation data (test_data.out) is that if we only build a dictionary based on training dataset (training_data.out), there may be some words in test_data.out which never appear in training_data.out. In this case, the generated "dict.pkl" is not the whole vacabulary. Considering it, I put both training data and test data to generate dict.pkl. As dict.pkl is consist of only token-id pairs, using test data will not affect the evaluation phase (no test info leak to model).
   
   If we don't want use test data in generating dict.pkl, we can change the command into this, to only use training data:
   
     $ python generate_dict.py -text_path1 'training_data.out' -text_path2 'training_data.out' -dict_path 'dict.pkl'



## Contact

Questions and discussion are welcome: vdthoang.2016@smu.edu.sg Or xinzhou.2020@phdcs.smu.edu.sg
