import time
import torch.nn as nn
from model.cpc_model1 import CPC
from model.classification_model import EmotionClassifier_cnn,EmotionClassifier_relu,EmotionClassifier_gru
from dataset import RawDownStreamDataset,RawDatasetMultipleFile


'''
up_stream --> intermediate task to learn the representation
down_stream --> actual task to be performed like classification, speaker detection.
test ---> predicting the output
'''
manual_seed=5
modes=["up_stream","down_stream_fine_tune","down_stream_train","test"]
mode_ISO=['us-tr','ds-ft','ds-tr','ds-ts']
mode_index=3
mode=modes[mode_index]
description='''
                CPC model training
            '''
experiment=None
time_string="_"+time.strftime("%d-%b_%H:%M:%S")
log_path="/scratch/kcprmo/cpc/CPC/experiments/"
use_cuda=True
timestep=12
batch_size=512
audio_window=128
channel=60
split_char="-"
split_position=2

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
run_name_prefix=None
run_name=None
training_history=None
ds_model_no_class=8
target_names=["neutral", "calm", "happy", "sad", "angry", "fearful","disgust", "surprised"]
"""
for puhelahjat data set the validation should be 4 percent extra to manage the 10k extra records.
"""
if mode==modes[0]:
    model=CPC #add the upstream model for pre-training
    train=True
    run_name_prefix=mode_ISO[mode_index]+"_"+model.__name__+"_"
    dataset=RawDatasetMultipleFile
    named_parameters_to_ignore=[]
    train_split=0.80 #only for puhelahjat
    data_list_path=None
    data_file_path=['/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_1.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_2.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_3.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_4.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_5.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_6.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_7.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_8.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_9.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_10.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_11.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_12.h5',
                    '/scratch/kcprmo/cpc/data/dataset/puhelahjat_14sec/puhelahjat_13.h5']
    data_list_path=['../data/dataset/libri_speech/dev-Librispeech.pkl','../data/dataset/libri_speech/train_100-Librispeech.pkl','../data/dataset/libri_speech/train_360-Librispeech.pkl','../data/dataset/libri_speech/train_500-Librispeech.pkl']
    data_file_path=['../data/dataset/libri_speech/dev-Librispeech.h5','../data/dataset/libri_speech/train_100-Librispeech.h5','../data/dataset/libri_speech/train_360-Librispeech.h5','../data/dataset/libri_speech/train_500-Librispeech.h5']
    #data_file_path=['/scratch/kcprmo/cpc/data/dataset/puhelahjat_1.h5',
    #                '/scratch/kcprmo/cpc/data/dataset/puhelahjat_2.h5',]
    #load_model="/scratch/kcprmo/cpc/CPC/experiments/models/test_CPC_puhelahjat_13_02-Apr_01:16:44-model_best.pth"
    load_model=None
    save_model=True
    train_ds_model=False
    validate_ds_model=False
    dataset_len=None
    epochs=50
    description='''
                CPC model pre-training with 960 hrs of librespeeh with learning rate 0.001
                '''
    lr = 1e-3
    is_model_load_strict=True


if mode==modes[1]:
    model=EmotionClassifier_cnn #add the ds_model_here
    #model=EmotionClassifier_gru
    dataset=RawDownStreamDataset
    run_name_prefix=mode_ISO[mode_index]+"_"+model.__name__+"_"
    named_parameters_to_ignore=['cpc']
    
    data_list_path="/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/train_english_emotion_speech.pkl"
    data_file_path='/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/english_emotion_speech.h5'
    dataset_len=None
    
    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=8
    ds_loss_fn = nn.CrossEntropyLoss()
    
    is_model_load_strict=False
    load_model='/scratch/kcprmo/cpc/CPC/experiments/models/us-tr_CPC_puhelahjat_56705mins_0.0001lr_23-Apr_13:03:44-model_best.pth'
    train_ds_model=True
    validate_ds_model=True
    save_model=True
    train=True
    lr = 1e-4
    batch_size=64
    epochs=200
    is_model_load_strict=False
    train_split=0.9
    description='''
                CPC model training the finish speech down stream task with the model pretrained on 960hrs of librispeech for 30 epoch and lr =0.001. Fine tuned with 100 epoch and 0.0001 LR.
            ''' 


if mode==modes[2]:
    model=EmotionClassifier_cnn #add the ds_model_here
    #model=EmotionClassifier_gru
    dataset=RawDownStreamDataset
    run_name_prefix=mode_ISO[mode_index]+"_"+model.__name__+"_"
    #run_name="em_classify_cnn_on_960hrs+fine_tuned_" + time_string
    named_parameters_to_ignore=[]
    
    data_list_path="/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/train_english_emotion_speech.pkl"
    data_file_path='/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/english_emotion_speech.h5'
    dataset_len=None

    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=8
    ds_loss_fn = nn.CrossEntropyLoss()
    
    is_model_load_strict=True
    load_model="/scratch/kcprmo/cpc/CPC/experiments/models/ds-ft_EmotionClassifier_cnn_puhelahjat_56705mins_0.0001lr_128AW_01-May_13:36:44-model_best.pth"
    train_ds_model=True
    validate_ds_model=True
    save_model=True
    train=True
    lr = 5e-4
    batch_size=64
    epochs=100
    train_split=0.9
    description='''
                CPC model training the finish speech down stream task with the model pretrained on 960hrs of librispeech for 30 epoch and lr =0.001. Fine tuned with 100 epoch and 0.0001 LR. \n
                Training the entire model on finnish data.
            ''' 

if mode==modes[3]:
    model=EmotionClassifier_cnn #add the ds_model_here
    #model=EmotionClassifier_gru #add the ds_model_here
    run_name_prefix=mode_ISO[mode_index]+"_"+model.__name__+"_"
    #run_name="test_" + time_string
    dataset=RawDownStreamDataset
    load_model='/scratch/kcprmo/cpc/CPC/experiments/models/ds-tr_EmotionClassifier_cnn_puhelahjat_56705mins_0.0001lr_128AW_01-May_13:39:21-model_best.pth'#original
    #load_model='/scratch/kcprmo/cpc/CPC/experiments/models/em_classify_cnn_on_960hrs+fine_tuned_14-Mar_00:15:28-model_best.pth'#base_line
    #load_model='/scratch/kcprmo/cpc/CPC/experiments/models/ds-tr_EmotionClassifier_gru_baseline_15-Mar_22:32:11-model_best.pth'#cpc_encoder+gru
    #load_model="/scratch/kcprmo/cpc/CPC/experiments/models/ds-tr_EmotionClassifier_cnn_libri-speech_58329mins_18-Apr_22:42:44-model_best.pth"
    data_list_path="/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/test_english_emotion_speech.pkl"
    data_file_path='/scratch/kcprmo/cpc/data/dataset/RAVDESS_english_emotion/english_emotion_speech.h5'
    dataset_len=None
    ds_model_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
    ds_model_no_class=8
    ds_model_loss_fn = nn.CrossEntropyLoss()
    validate_ds_model=True
    test=True
    split_data=False
    is_model_load_strict=True
 
 
