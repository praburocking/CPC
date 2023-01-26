import time
audio_window=2048
use_cuda=False
timestep=12
batch_size=64
audio_window=20480
warmup_steps=20
logging_dir='./logs'
epochs=1
train_split=0.4
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
training_mode="down_stream"
#training_mode=="up_stream"
dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'