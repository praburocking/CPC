import numpy as np
import torch
from torch.utils import data
import h5py
from scipy.io import wavfile
from collections import defaultdict
from random import randint
import pickle
    
class RawDatasetMultipleFile(data.Dataset):
    def __init__(self, raw_files, list_files, audio_window, dataset_len=None,split_char="_",split_position=-1):
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
        total_len=0
        if list_files is not None:
            for list_file,raw_file in zip(list_files,raw_files):
                with open(list_file, "rb") as fp:   # Unpickling
                    temp=pickle.load(fp)
                temp = [x.strip() for x in temp]
                self.temp_file_lists.extend(temp)
                h5f = h5py.File(raw_file, 'r')
                for i in temp: # sanity check
                    frame_len = h5f[i].shape[1]
                    if frame_len > audio_window:
                        total_len+=frame_len
                        self.valid_log_mel_list.append(i)
                    if dataset_len is not None and dataset_len <= len(self.valid_log_mel_list):
                        print("breaking as the dataset_len reached")
                        break
                self.h5files.append(h5f)
                if dataset_len is not None and dataset_len <= len(self.valid_log_mel_list):
                    print("breaking as the dataset_len reached")
                    break
                
        else:
            for raw_file in raw_files:
                h5f = h5py.File(raw_file, 'r')
                temp=h5f.keys()
                self.temp_file_lists.extend(temp)
                for i in temp: # sanity check
                    frame_len = h5f[i].shape[1]
                    if frame_len > audio_window:
                        total_len+=frame_len
                        self.valid_log_mel_list.append(i)
                    if dataset_len is not None and dataset_len <= len(self.valid_log_mel_list):
                        print("breaking as the dataset_len reached")
                        break
                self.h5files.append(h5f)
                if dataset_len is not None and dataset_len <= len(self.valid_log_mel_list):
                    print("breaking as the dataset_len reached")
                    break
        self.total_audio_in_mins=total_len/(100*60)
        print("the total size of the audio in mins is --- "+str(self.total_audio_in_mins))
        print("dataset_len ----- "+str(dataset_len))

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
    def __init__(self, raw_file, list_file, audio_window,dataset_len=None,split_char="_",split_position=-1):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 30
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []
        self.total_audio_in_mins=None
        self.split_char=split_char
        self.split_position=split_position
        total_len=0

        with open(list_file, "rb") as fp:   # Unpickling
            temp = pickle.load(fp)
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[1]
            if utt_len > self.audio_window:
                total_len+=utt_len
                self.utts.append(i)
        self.total_audio_in_mins=total_len/(100*60)
        print("the total size of the audio in mins is --- "+str(self.total_audio_in_mins))
        print("total number of audios ---- "+str(len(self.utts)))
        print("dataset_len ----- "+str(dataset_len))
                

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        #print(index)
        utt_id = self.utts[index] # get the utterance id
        #print(utt_id)
        y=utt_id.split(self.split_char)[self.split_position]
        #print(y)
        #print(utt_id+"-----"+str(y))
        utt_len = self.h5f[utt_id].shape[1] # get the number of data points in the utterance
        win_index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        y2int=lambda x:int(x)-1
        #print("win index extracted")
        return self.h5f[utt_id][:,win_index:win_index+self.audio_window],y2int(y)

def get_dataloaders(conf,tensor_writer=None):

    assert conf.dataset is not None, "conf.dataset cannot be None" 
    
    dataset=conf.dataset(conf.data_file_path, conf.data_list_path, conf.audio_window,dataset_len=conf.dataset_len,split_char=conf.split_char,split_position=conf.split_position)
    if tensor_writer is not None:
        tensor_writer.add_text('total_data_size',str(dataset.total_audio_in_mins)+" mins :: len of data "+str(len(dataset)) )
        #tensor_writer.add_text('total_train_data_size_in_mins',str(validation_loader.dataset.total_audio_in_mins))
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


