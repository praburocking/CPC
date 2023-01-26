import soundfile as sf
import os 
import h5py
import numpy as np
import pickle



def audio2raw(rootdirs,extension,outputfile_name,outputlist_name):

    h5f = h5py.File(outputfile_name, 'w')
    file_names=[]
    for rootdir in rootdirs:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith(extension):
                    fullpath = os.path.join(subdir, file)
                    data,fs = sf.read(fullpath)
                    data=np.array(data)
                    h5f.create_dataset(file[:-len(extension)], data=data)
                    file_names.append(file[:-len(extension)])
                    print(file[:-len(extension)])
    with open(outputlist_name, "wb") as fp: 
        pickle.dump(file_names,fp)
    h5f.close()

dev_outputfile_name='dev-Librispeech.h5'
dev_outputlist_name='dev-Librispeech.pkl'
dev_rootdirs=['../data/LibriSpeech/dev-clean/','../data/LibriSpeech/dev-other/']
extension=".flac"
audio2raw(dev_rootdirs,extension,dev_outputfile_name,dev_outputlist_name)

test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
test_rootdirs=['../data/LibriSpeech/test-clean/','../data/LibriSpeech/test-other/']
audio2raw(test_rootdirs,extension,test_outputfile_name,test_outputlist_name)


finnish_speech_outputfile_name='finnish_speech.h5'
finnish_speech_outputlist_name='finnish_speech.pkl'
finnish_speech_rootdirs=['../data/FESC_segmented/']
finnish_speech_extension='.WAV'
audio2raw(finnish_speech_rootdirs,finnish_speech_extension,finnish_speech_outputfile_name,finnish_speech_outputlist_name)
