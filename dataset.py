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
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file, "rb") as fp:   # Unpickling
            temp = pickle.load(fp)
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return self.h5f[utt_id][index:index+self.audio_window]


class RawDownStreamDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file, "rb") as fp:   # Unpickling
            temp = pickle.load(fp)
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id
        
        y=utt_id.split("_")[-1] 
        print(utt_id+"-----"+str(y))
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return self.h5f[utt_id][index:index+self.audio_window],y

def get_dataloaders(conf):
    if conf.training_mode=="up_stream":
        training_set   = RawDataset(conf.dev_outputfile_name, conf.dev_outputlist_name, conf.audio_window)
        no_training_data=int(len(training_set)*conf.train_split)
        no_val_data=int(len(training_set)-no_training_data)

        test_set   = RawDataset(conf.test_outputfile_name, conf.test_outputlist_name, conf.audio_window)
        training_set, validation_set = torch.utils.data.random_split(training_set, [no_training_data, no_val_data])

        train_loader = data.DataLoader(training_set, batch_size=conf.batch_size, shuffle=True)
        validation_loader = data.DataLoader(validation_set, batch_size=conf.batch_size, shuffle=False)
        test_loader = data.DataLoader(test_set, batch_size=conf.batch_size, shuffle=False)

        return train_loader,validation_loader,test_loader
    elif conf.training_mode=="down_stream":
        finnish_speech_training_set   = RawDownStreamDataset(conf.finnish_speech_outputfile_name, conf.finnish_speech_outputlist_name, conf.audio_window)
        no_training_data=int(len(finnish_speech_training_set)*conf.train_split)
        no_val_data=int(len(finnish_speech_training_set)-no_training_data)
        finnish_speech_training_set, finnish_speech_validation_set = torch.utils.data.random_split(finnish_speech_training_set, [no_training_data, no_val_data])
        finnish_train_loader = data.DataLoader(finnish_speech_training_set, batch_size=conf.batch_size, shuffle=True)
        finnish_validation_loader = data.DataLoader(finnish_speech_validation_set, batch_size=conf.batch_size, shuffle=False)
        return finnish_train_loader,finnish_validation_loader,None
    
    


