import numpy as np
import torch
from torch.utils import data
import h5py
from scipy.io import wavfile
from collections import defaultdict
from random import randint
import pickle


class RawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 30
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file, "rb") as fp:   # Unpickling
            temp = pickle.load(fp)
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[1]
            if utt_len > audio_window:
                self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        mel_len = self.h5f[utt_id].shape[1] # get the number of data points in the utterance
        index = np.random.randint(mel_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return self.h5f[utt_id][:,index:index+self.audio_window],[]
    
    
class RawDatasetMultipleFile(data.Dataset):
    def __init__(self, raw_files, list_files, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 30
        """
        self.raw_files  = raw_files 
        self.audio_window = audio_window 
        self.list_files=list_files
        self.valid_log_mel_list = []
        self.temp_file_lists=[]
        self.h5files=[]
        for list_file,raw_file in zip(list_files,raw_files):
            with open(list_file, "rb") as fp:   # Unpickling
                temp=pickle.load(fp)
            temp = [x.strip() for x in temp]
            self.temp_file_lists.extend(temp)
            h5f = h5py.File(raw_file, 'r')
            for i in temp: # sanity check
                frame_len = h5f[i].shape[1]
                if frame_len > audio_window:
                    self.valid_log_mel_list.append(i)
            self.h5files.append(h5f)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.valid_log_mel_list)

    def __getitem__(self, index):
        id = self.valid_log_mel_list[index] # get the utterance id
        log_mel=None
        
        for h5f in self.h5files:
            try:
                log_mel=h5f[id]
                break  
            except KeyError:
                pass
                       
        mel_len = log_mel.shape[1] # get the number of data points in the utterance
        index = np.random.randint(mel_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return log_mel[:,index:index+self.audio_window],[]
    
class RawDatasetMultipleFileV1(data.Dataset):
    def __init__(self, raw_files,list_files, audio_window):
        """ raw_file: train-clean-100.h5
        """
        self.raw_files  = raw_files 
        self.audio_window = audio_window 
        self.valid_log_mel_list = []
        self.temp_file_lists=[]
        self.h5files=[]
        for raw_file in raw_files:
            
            h5f = h5py.File(raw_file, 'r')
            temp=h5f.keys()
            self.temp_file_lists.extend(temp)
            for i in temp: # sanity check
                frame_len = h5f[i].shape[1]
                if frame_len > audio_window:
                    self.valid_log_mel_list.append(i)
            self.h5files.append(h5f)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.valid_log_mel_list)

    def __getitem__(self, index):
        id = self.valid_log_mel_list[index] # get the utterance id
        log_mel=None
        
        for h5f in self.h5files:
            try:
                log_mel=h5f[id]
                break  
            except KeyError:
                pass
                       
        mel_len = log_mel.shape[1] # get the number of data points in the utterance
        index = np.random.randint(mel_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return log_mel[:,index:index+self.audio_window],[]


class RawDownStreamDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 30
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file, "rb") as fp:   # Unpickling
            temp = pickle.load(fp)
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[1]
            if utt_len > self.audio_window:
                self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id
        
        y=utt_id.split("_")[-1]
        #print(utt_id+"-----"+str(y))
        utt_len = self.h5f[utt_id].shape[1] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        y2int=lambda x:int(x)-1
        return self.h5f[utt_id][:,index:index+self.audio_window],y2int(y)

def get_dataloaders(conf):

    assert conf.dataset is not None, "conf.dataset cannot be None" 
    
    dataset=conf.dataset(conf.data_file_path, conf.data_list_path, conf.audio_window)
    print(conf.data_file_path)
    print(conf.data_list_path)
    print("len of dataset "+str(len(dataset)))
    if conf.split_data and conf.train_split is not None:
        no_training_data=int(len(dataset)*conf.train_split)
        no_val_data=int(len(dataset)-no_training_data)

        training_set, validation_set = torch.utils.data.random_split(dataset, [no_training_data, no_val_data])
        print(training_set)
        
        train_loader = data.DataLoader(training_set, batch_size=conf.batch_size, shuffle=True,drop_last=True)
        validation_loader = data.DataLoader(validation_set, batch_size=conf.batch_size, shuffle=True)

        return train_loader,validation_loader
    else:
            return data.DataLoader(dataset, batch_size=conf.batch_size, shuffle=True,drop_last=False)


