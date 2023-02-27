import soundfile as sf
import os 
import h5py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split



def audio2raw(rootdirs,extension,outputfile_name,outputlist_name):

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
                    h5f.create_dataset(file[:-len(extension)], data=data)
                    file_names.append(file[:-len(extension)])
                    print(file[:-len(extension)])
    with open(outputlist_name, "wb") as fp: 
        pickle.dump(file_names,fp)
    h5f.close()

def perform_train_test_split(input_list_path,train_list_path,test_list_path):
        with open(input_list_path, "rb") as fp:   # Unpickling
            file_list = pickle.load(fp)
        file_list = [x.strip() for x in file_list]

        str_int=lambda x:int(x.split("_")[-1])-1
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
#dev_rootdirs=['../data/LibriSpeech/dev-clean/','../data/LibriSpeech/dev-other/','../raw_data/LibriSpeech/train-clean-100/','../raw_data/LibriSpeech/train-clean-360/','../raw_data/LibriSpeech/train-other-500/']
extension=".flac"
audio2raw(dev_rootdirs,extension,dev_outputfile_name,dev_outputlist_name)


train_100_outputfile_name='train_100-Librispeech.h5'
train_100_outputlist_name='train_100-Librispeech.pkl'
train_100_rootdirs=['../raw_data/LibriSpeech/train-clean-100/']
extension=".flac"
audio2raw(train_100_rootdirs,extension,train_100_outputfile_name,train_100_outputlist_name)

train_360_outputfile_name='train_360-Librispeech.h5'
train_360_outputlist_name='train_360-Librispeech.pkl'
train_360_rootdirs=['../raw_data/LibriSpeech/train-clean-360/']
extension=".flac"
audio2raw(train_360_rootdirs,extension,train_360_outputfile_name,train_360_outputlist_name)


train_500_outputfile_name='train_500-Librispeech.h5'
train_500_outputlist_name='train_500-Librispeech.pkl'
train_500_rootdirs=['../raw_data/LibriSpeech/train-other-500/']
extension=".flac"
audio2raw(train_500_rootdirs,extension,train_500_outputfile_name,train_500_outputlist_name)

test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
test_rootdirs=['../data/LibriSpeech/test-clean/','../data/LibriSpeech/test-other/']
#audio2raw(test_rootdirs,extension,test_outputfile_name,test_outputlist_name)


finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'
train_finnish_speech_outputlist_name="train_finnish_speech.pkl"
test_finnish_speech_outputlist_name="test_finnish_speech.pkl"
finnish_speech_rootdirs=['../data/FESC_segmented/']
finnish_speech_extension='.WAV'
#audio2raw(finnish_speech_rootdirs,finnish_speech_extension,finnish_speech_outputfile_name,finnish_speech_outputlist_name)
#perform_train_test_split(finnish_speech_outputlist_name,train_finnish_speech_outputlist_name,test_finnish_speech_outputlist_name)



