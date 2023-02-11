import time
import torch.nn as nn
audio_window=2048
use_cuda=False
timestep=12
batch_size=16
audio_window=20480
warmup_steps=20
logging_dir='./logs'
epochs=1
train_split=0.2
run_name = "cdc_" + time.strftime("-%Y-%m-%d_%H_%M_%S")
run_name_us_model = "cdc_" + time.strftime("-%Y-%m-%d_%H_%M_%S")
run_name_ds_model = "class_" + time.strftime("-%Y-%m-%d_%H_%M_%S")

'''
up_stream --> intermediate task to learn the representation
down_stream --> actual task to be performed like classification, speaker detection.
'''
training_mode="down_stream"
#training_mode="up_stream"
dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'
emotion_classifier_linear_config=[{"in_dim":512,"out_dim":256},{"in_dim":256,"out_dim":124},{"in_dim":124,"out_dim":64}]
emotion_classifier_no_class=5
load_model=True
model_path='.\logs\CPC-model_best.pth'
down_stream_loss_fn = nn.CrossEntropyLoss()