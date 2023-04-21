import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import soundfile as sf
import os 
import h5py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import librosa



N_MELS=60
def audio2raw(rootdirs,extension,outputfile_name,outputlist_name,is_change_to_log_mel=True):

    h5f = h5py.File(outputfile_name, 'w')
    file_names=[]
    for rootdir in rootdirs:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith(extension):
                    fullpath = os.path.join(subdir, file)
                    data,fs = sf.read(fullpath)
                    data=np.array(data, dtype='float32')
                    #print(data.dtype)
                    if len(data.shape)==2:
                        data=np.array(data[:,0])
                    num_fft = int(0.03 * fs)
                    shift = int(0.01 * fs)
                    mel_spectrogram = librosa.feature.melspectrogram(data, sr=fs, n_fft=num_fft, hop_length=shift, n_mels=N_MELS)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                    #print(log_mel_spectrogram.shape)
                    #print(type(log_mel_spectrogram))
                    #print(log_mel_spectrogram.dtype)
                    # mfccs = librosa.feature.mfcc(data, sr=sr,n_mfcc=40)
                    h5f.create_dataset(file[:-len(extension)], data=log_mel_spectrogram)
                    file_names.append(file[:-len(extension)])
                    print(file[:-len(extension)])
    with open(outputlist_name, "wb") as fp: 
        pickle.dump(file_names,fp)
    h5f.close()

def perform_train_test_split(input_list_path,train_list_path,test_list_path,seperator="_",position=-1):
        with open(input_list_path, "rb") as fp:   # Unpickling
            file_list = pickle.load(fp)
        file_list = [x.strip() for x in file_list]

        str_int=lambda x:int(x.split(seperator)[position])-1
        apply_fun=np.vectorize(str_int)
        class_list=apply_fun(file_list)
        print(str(class_list))

        train_file_list,test_file_list=train_test_split(file_list, test_size=0.20, random_state=42,shuffle=True, stratify=class_list)
        with open(train_list_path, "wb") as fp: 
            pickle.dump(train_file_list,fp)
        with open(test_list_path, "wb") as fp: 
            pickle.dump(test_file_list,fp)
       


dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
dev_rootdirs=['../data/LibriSpeech/dev-clean/','../data/LibriSpeech/dev-other/']
extension=".flac"
#audio2raw(dev_rootdirs,extension,dev_outputfile_name,dev_outputlist_name)


train_100_outputfile_name='train_100-Librispeech.h5'
train_100_outputlist_name='train_100-Librispeech.pkl'
train_100_rootdirs=['../raw_data/LibriSpeech/train-clean-100/']
extension=".flac"
#audio2raw(train_100_rootdirs,extension,train_100_outputfile_name,train_100_outputlist_name)

train_360_outputfile_name='train_360-Librispeech.h5'
train_360_outputlist_name='train_360-Librispeech.pkl'
train_360_rootdirs=['../raw_data/LibriSpeech/train-clean-360/']
extension=".flac"
#audio2raw(train_360_rootdirs,extension,train_360_outputfile_name,train_360_outputlist_name)


train_500_outputfile_name='train_500-Librispeech.h5'
train_500_outputlist_name='train_500-Librispeech.pkl'
train_500_rootdirs=['../raw_data/LibriSpeech/train-other-500/']
extension=".flac"
audio2raw(train_500_rootdirs,extension,train_500_outputfile_name,train_500_outputlist_name)

test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
test_rootdirs=['../data/LibriSpeech/test-clean/','../data/LibriSpeech/test-other/']
audio2raw(test_rootdirs,extension,test_outputfile_name,test_outputlist_name)


finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'
train_finnish_speech_outputlist_name="train_finnish_speech.pkl"
test_finnish_speech_outputlist_name="test_finnish_speech.pkl"
finnish_speech_rootdirs=['../data/FESC_segmented/']
finnish_speech_extension='.WAV'
audio2raw(finnish_speech_rootdirs,finnish_speech_extension,finnish_speech_outputfile_name,finnish_speech_outputlist_name)
perform_train_test_split(finnish_speech_outputlist_name,train_finnish_speech_outputlist_name,test_finnish_speech_outputlist_name)


english_speech_outputfile_name='english_emotion_speech.h5'
english_speech_outputlist_name='english_emotion_speech.pkl'
english_emotion_speech_rootdirs=['/scratch/kcprmo/cpc/raw_data/English_emotion']
english_speech_extension='.wav'
train_english_speech_outputlist_name='train_english_emotion_speech.pkl'
test_english_speech_outputlist_name='test_english_emotion_speech.pkl'
audio2raw(english_emotion_speech_rootdirs,english_speech_extension,english_speech_outputfile_name,english_speech_outputlist_name)
perform_train_test_split(english_speech_outputlist_name,train_english_speech_outputlist_name,test_english_speech_outputlist_name,seperator="-",position=2)

