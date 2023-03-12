import time
import torch.nn as nn
from model.cpc_model1 import CPC
from model.classification_model import EmotionClassifier_cnn,EmotionClassifier_relu
from dataset import RawDataset,RawDownStreamDataset,RawDatasetMultipleFile


'''
up_stream --> intermediate task to learn the representation
down_stream --> actual task to be performed like classification, speaker detection.
test ---> predicting the output
'''
modes=["up_stream","down_stream_fine_tune","down_stream_train","test"]
mode=modes[3]
description='''
                CPC model training
            '''
time_string=time.strftime("%d-%b_%H:%M:%S")
log_path="/scratch/kcprmo/cpc/CPC/experiments/"
use_cuda=True
timestep=12
batch_size=1024
audio_window=128
channel=60

epochs=15
train_split=0.8
log_interval=5
lr = 1e-3
patience_thresold=5


emotion_classifier_linear_config=None
ds_model_no_class=None
ds_model_loss_fn = None
model=None
named_parameters_to_ignore=[]
load_model=None
train_ds_model=False
validate_ds_model=False
save_model=False
dataset=None
train=False
test=False
split_data=True



if mode==modes[0]:
    model=CPC #add the upstream model for pre-training
    dataset=RawDatasetMultipleFile
    run_name="cpc_train_dev960hr_" + time_string
    named_parameters_to_ignore=[]
    data_list_path=['../data/dataset/dev-Librispeech.pkl','../data/dataset/train_100-Librispeech.pkl','../data/dataset/train_360-Librispeech.pkl','../data/dataset/train_500-Librispeech.pkl']
    data_file_path=['../data/dataset/dev-Librispeech.h5','../data/dataset/train_100-Librispeech.h5','../data/dataset/train_360-Librispeech.h5','../data/dataset/train_500-Librispeech.h5']
    load_model=None
    save_model=True
    train_ds_model=False
    validate_ds_model=False
    train=True
    epochs=30
    description='''
                CPC model pre-training with 960 hrs of librespeeh with learning rate 0.001
                '''
    lr = 1e-3
    is_model_load_strict=True


if mode==modes[1]:
    model=EmotionClassifier_cnn #add the ds_model_here
    dataset=RawDownStreamDataset
    run_name="em_classify_cnn_on_960hrs_" + time_string
    named_parameters_to_ignore=['cpc']
    
    data_list_path="../data/dataset/train_finnish_speech.pkl"
    data_file_path='../data/dataset/finnish_speech.h5'

    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=5
    ds_loss_fn = nn.CrossEntropyLoss()
    
    is_model_load_strict=False
    load_model='/scratch/kcprmo/cpc/CPC/experiments/models/cpc_train_dev960hr_11-Mar_01:32:19-model_best.pth'
    train_ds_model=True
    validate_ds_model=True
    save_model=True
    train=True
    lr = 1e-4
    batch_size=1024
    epochs=200
    is_model_load_strict=False
    description='''
                CPC model training the finish speech down stream task with the model pretrained on 960hrs of librispeech for 30 epoch and lr =0.001. Fine tuned with 100 epoch and 0.0001 LR.
            ''' 


if mode==modes[2]:
    model=EmotionClassifier_cnn #add the ds_model_here
    dataset=RawDownStreamDataset
    run_name="em_classify_cnn_on_960hrs+fine_tuned_" + time_string
    named_parameters_to_ignore=[]
    
    data_list_path="../data/dataset/train_finnish_speech.pkl"
    data_file_path='../data/dataset/finnish_speech.h5'

    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=5
    ds_loss_fn = nn.CrossEntropyLoss()
    
    is_model_load_strict=True
    load_model='/scratch/kcprmo/cpc/CPC/experiments/models/em_classify_cnn_on_960hrs_12-Mar_11:52:07-model_best.pth'
    train_ds_model=True
    validate_ds_model=True
    save_model=True
    train=True
    lr = 5e-4
    batch_size=64
    epochs=100
    description='''
                CPC model training the finish speech down stream task with the model pretrained on 960hrs of librispeech for 30 epoch and lr =0.001. Fine tuned with 100 epoch and 0.0001 LR. \n
                Training the entire model on finnish data.
            ''' 

if mode==modes[3]:
    model=EmotionClassifier_cnn #add the ds_model_here
    run_name="test_" + time_string
    dataset=RawDownStreamDataset
    load_model='/scratch/kcprmo/cpc/CPC/experiments/models/em_classify_cnn_on_960hrs+fine_tuned_12-Mar_13:31:55-model_best.pth'
    data_list_path="../data/dataset/test_finnish_speech.pkl"
    data_file_path='../data/dataset/finnish_speech.h5'
    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=5
    ds_model_loss_fn = nn.CrossEntropyLoss()
    validate_ds_model=True
    test=True
    split_data=False
    is_model_load_strict=True
 