import time
import torch.nn as nn
from model.cpc_model1 import CPC
from model.classification_model import EmotionClassifier
from dataset import RawDataset,RawDownStreamDataset,RawDatasetMultipleFile


'''
up_stream --> intermediate task to learn the representation
down_stream --> actual task to be performed like classification, speaker detection.
test ---> predicting the output
'''
modes=["up_stream","down_stream","test"]
mode=modes[0]
description='''
                CPC model training
            '''

use_cuda=True
timestep=12
batch_size=1024
audio_window=128
warmup_steps=20
logging_dir='./logs'
epochs=15
train_split=0.8
log_interval=5
lr = 1e-3
patience_thresold=5
test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'


emotion_classifier_linear_config=None
ds_model_no_class=None
ds_model_loss_fn = None
us_model=None
ds_model=None
named_parameters_to_ignore=[]
load_us_model=None
load_ds_model=None
train_us_model=False
train_ds_model=False
validate_us_model=False
validate_ds_model=False
save_us_model=False
save_ds_model=False
dataset=None
train=False
test=False
split_data=True


if mode==modes[0]:
    us_model=CPC #add the us_model_here
    dataset=RawDatasetMultipleFile
    run_name="cpc_train_dev_960hr_" + time.strftime("%m-%d_%H_%M_%S")
    named_parameters_to_ignore=[]
    data_list_path=['dev-Librispeech.pkl','train_100-Librispeech.pkl','train_360-Librispeech.pkl','train_500-Librispeech.pkl']
    data_file_path=['dev-Librispeech.h5','train_100-Librispeech.h5','train_360-Librispeech.h5','train_500-Librispeech.h5']
    load_us_model='./logs/cpc_train_dev_100_03-06_00_42_00-model_best.pth'
    save_us_model=True
    train_us_model=True
    validate_us_model=True
    train=True
    epochs=30
    description='''
                CPC model pre-training with 960 hrs of librespeeh with learning rate 0.001
                '''
    lr = 1e-3


if mode==modes[1]:
    us_model=CPC #add the us_model_here
    ds_model=EmotionClassifier #add the ds_model_here
    dataset=RawDownStreamDataset
    run_name="Em_classify_on_960hrs_" + time.strftime("%m-%d_%H_%M_%S")
    named_parameters_to_ignore=['cpc']
    
    data_list_path="train_finnish_speech.pkl"
    data_file_path='finnish_speech.h5'

    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=5
    ds_model_loss_fn = nn.CrossEntropyLoss()
    
    load_us_model='/scratch/kcprmo/cpc/CPC/logs/cpc_train_dev_100_360_500_03-06_10_41_59-model_best.pth'
    train_us_model=True
    train_ds_model=True
    validate_us_model=True
    validate_ds_model=True
    save_ds_model=True
    train=True
    description='''
                CPC model training the finish speech down stream task with the model pretrained on 960hrs of librispeech for 25 epoch and lr =0.0001.
            ''' 

if mode==modes[2]:
    us_model=CPC #add the us_model_here
    ds_model=EmotionClassifier #add the ds_model_here
    run_name="test_" + time.strftime("%m-%d_%H_%M_%S")
    dataset=RawDownStreamDataset
    load_us_model='/scratch/kcprmo/cpc/CPC/logs/cdc_02-24_17_06_52-model_best.pth'
    load_ds_model='/scratch/kcprmo/cpc/CPC/logs/EmotionClassifier_02-24_17_16_00-model_best.pth'
    data_list_path="test_finnish_speech.pkl"
    data_file_path='finnish_speech.h5'
    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=5
    ds_model_loss_fn = nn.CrossEntropyLoss()
    validate_us_model=True
    validate_ds_model=True
    test=True
    split_data=False
 