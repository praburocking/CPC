import time
import torch.nn as nn
use_cuda=True
timestep=12
batch_size=128
audio_window=20480
warmup_steps=20
logging_dir='./logs'
epochs=10
train_split=0.8
run_name = "trail_run_" + time.strftime("%Y-%m-%d_%H_%M_%S")
run_name_us_model = "cdc_" + time.strftime("%Y-%m-%d_%H_%M_%S")
run_name_ds_model = "EmotionClassifier_" + time.strftime("%Y-%m-%d_%H_%M_%S")
patience_thresold=5

'''
up_stream --> intermediate task to learn the representation
down_stream --> actual task to be performed like classification, speaker detection.
test ---> predicting the output
'''
mode="up_stream"
#mode="down_stream" 
#mode="test"
#mode="none"
dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'
train_finnish_speech_outputlist_name="train_finnish_speech.pkl"
test_finnish_speech_outputlist_name="test_finnish_speech.pkl"
emotion_classifier_linear_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
emotion_classifier_no_class=5
load_model=True
us_model_path='/scratch/kcprmo/cpc/CPC/logs/cdc_-2023-02-21_21_40_54-model_best.pth'
ds_model_path='/scratch/kcprmo/cpc/CPC/logs/EmotionClassifier_-2023-02-22_08_12_38__-2023-02-22_10_24_37-model_best.pth'
down_stream_loss_fn = nn.CrossEntropyLoss()