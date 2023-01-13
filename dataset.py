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
