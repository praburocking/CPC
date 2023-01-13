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
dev_rootdirs=['../dataset/LibriSpeech/dev-clean/','../dataset/LibriSpeech/dev-other/']
extension=".flac"
audio2raw(dev_rootdirs,extension,dev_outputfile_name,dev_outputlist_name)

test_outputfile_name='test-Librispeech.h5'
test_outputlist_name='test-Librispeech.pkl'
test_rootdirs=['../dataset/LibriSpeech/test-clean/','../dataset/LibriSpeech/test-other/']
audio2raw(test_rootdirs,extension,test_outputfile_name,test_outputlist_name)

