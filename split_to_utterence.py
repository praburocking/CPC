import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import soundfile as sf
from pydub import AudioSegment
import soundfile as sf
import os 
import h5py
import numpy as np
import pickle
import pandas as pd
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
from sklearn.model_selection import train_test_split
import librosa

from datetime import datetime

dialect_classes=['Pirkanmaa', 'Pohjois-Pohjanmaa', 'Pohjois-Savo',
'Varsinais-Suomi', 'Keski-Suomi', 'Häme', 'Satakunta','Pohjois-Karjala']



N_MELS=60
def audio2raw(base_dir,meta_datas,rootdirs,extension,outputfile_name,outputlist_name,is_split_on_silence=False):


    file_names=[]
    class_names=[]
    chunk_length=13700
    chunk_length=5000
    slience_in_ms=300
    total_file_not_found=0
    print(rootdirs)
    audio_size=0
    for i,rootdir in enumerate(rootdirs):
        if (i+1)<8:
            print("skipping ... "+str(i+1)+":: csv..."+meta_datas[i]+" :: dir..."+rootdir)
            #continue
        meta_file_path=os.path.join(base_dir, meta_datas[i])
        meta_data=pd.read_csv(meta_file_path)
        temp_output_filename=outputfile_name+"_"+str(i+1)+".h5"
        temp_output_csv_filename=outputlist_name+"_"+str(i+1)+".csv"
        h5f = h5py.File(temp_output_filename, 'w')
        
        columns=meta_data.columns.tolist()
        columns=columns.append("length_ms")
        output_file=pd.DataFrame(columns =columns )
            
        for index, row in meta_data.iterrows():
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            if row["dialect"] not in dialect_classes:
                continue
            file_name=row["recordingId"]+".flac"
            cur_dir=os.path.join(base_dir, rootdir)
            fullpath = os.path.join(cur_dir, file_name)
            if (os.path.isfile(fullpath)):
                flac = AudioSegment.from_file(fullpath, format='flac')
                audio_size=flac.duration_seconds+audio_size
                #audio_chunks = split_on_silence(flac, min_silence_len=1000,silence_thresh=-16)
                audio_chunks=None
                if is_split_on_silence:
                    audio_chunks=split_on_silence(flac, min_silence_len=slience_in_ms)
                else:
                    audio_chunks = make_chunks(flac, chunk_length)
                    
                for j, chunk in enumerate(audio_chunks):
                    output_name=row["recordingId"]+"_"+str(j)
                    output_name=str(dialect_classes.index(row["dialect"])+1)+"_"+output_name # for dialect classification
                    samples = chunk.get_array_of_samples()
                    fs=chunk.frame_rate
                    data = np.array(samples).T.astype(np.float32)
                    num_fft = int(0.03 * fs)
                    shift = int(0.01 * fs)
                    mel_spectrogram = librosa.feature.melspectrogram(data, sr=fs, n_fft=num_fft, hop_length=shift, n_mels=N_MELS)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                    h5f.create_dataset(output_name, data=log_mel_spectrogram)
                    #file_handle = chunk.export(outputfile_name+output_name+".wav", format="wav")
                    temp_row=row.copy()
                    temp_row["recordingId"]=output_name
                    temp_row["length_ms"]=len(chunk)
                    output_file=pd.concat([output_file,temp_row.to_frame().T],ignore_index=True)
                    print("file created -- "+output_name)
            else:
                print(current_time+" :: file not found "+str(file_name),flush=True)
                total_file_not_found+=1
        output_file.to_csv(temp_output_csv_filename)
    print(total_file_not_found)
    print("audio size  in mins "+str(audio_size/60))
    h5f.close()


def perform_train_test_split(input_list_path,train_list_path,test_list_path,seperator="_",position=-1):
        #h5f = h5py.File(input_list_path+".h5", 'r')
        #file_list=list(h5f.keys())
        with open(input_list_path, "rb") as fp:   # Unpickling
            file_list = pickle.load(fp)
        file_list = [x.strip() for x in file_list]

        str_int=lambda x:int(x.split(seperator)[position])-1
        apply_fun=np.vectorize(str_int)
        class_list=apply_fun(file_list)
        print(str(class_list))

        train_file_list,test_file_list=train_test_split(file_list, test_size=0.15, random_state=42,shuffle=True, stratify=class_list)
        with open(train_list_path, "wb") as fp: 
            pickle.dump(train_file_list,fp)
        with open(test_list_path, "wb") as fp: 
            pickle.dump(test_file_list,fp)
                    

finnish_speech_outputfile_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_slient_300ms/puhelahjat'
finnish_speech_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_slient_300ms/puhelahjat_meta'
base_dir="/scratch/kcprmo/cpc/finnish_data/puhelahjat/v1/"
#finnish_speech_rootdirs=["2020_part_01/audio_annotated","2020_part_02/audio_annotated","2020_part_03/audio_annotated","2020_part_04/audio_annotated","2020_part_05/audio_annotated","2020_part_06/audio_annotated","2020_part_07/audio_annotated","2020_part_08/audio_annotated","2020_part_09/audio_annotated"]
#meta_datas=["2020_part_01/puhelahjat_2020_part_01_meta.csv","2020_part_02/puhelahjat_2020_part_02_meta.csv","2020_part_03/puhelahjat_2020_part_03_meta.csv","2020_part_04/puhelahjat_2020_part_04_meta.csv","2020_part_05/puhelahjat_2020_part_05_meta.csv","2020_part_06/puhelahjat_2020_part_06_meta.csv","2020_part_07/puhelahjat_2020_part_07_meta.csv","2020_part_08/puhelahjat_2020_part_08_meta.csv","2020_part_09/puhelahjat_2020_part_09_meta.csv"]
finnish_speech_rootdirs=["2020_part_01/audio_annotated","2020_part_02/audio_annotated","2020_part_03/audio_annotated","2020_part_04/audio_annotated","2020_part_05/audio_annotated","2020_part_06/audio_annotated","2020_part_07/audio_annotated","2020_part_08/audio_annotated","2020_part_09/audio_annotated"]
meta_datas=["2020_part_01/puhelahjat_2020_part_01_meta.csv","2020_part_02/puhelahjat_2020_part_02_meta.csv","2020_part_03/puhelahjat_2020_part_03_meta.csv","2020_part_04/puhelahjat_2020_part_04_meta.csv","2020_part_05/puhelahjat_2020_part_05_meta.csv","2020_part_06/puhelahjat_2020_part_06_meta.csv","2020_part_07/puhelahjat_2020_part_07_meta.csv","2020_part_08/puhelahjat_2020_part_08_meta.csv","2020_part_09/puhelahjat_2020_part_09_meta.csv"]
finnish_speech_rootdirs.extend(["2020_part_10/audio_annotated","2020_part_11/audio_annotated","2020_part_12/audio_annotated","2020_part_13/audio_annotated"])
meta_datas.extend(["2020_part_10/puhelahjat_2020_part_10_meta.csv","2020_part_11/puhelahjat_2020_part_11_meta.csv","2020_part_12/puhelahjat_2020_part_12_meta.csv","2020_part_13/puhelahjat_2020_part_13_meta.csv"])
finnish_speech_extension='.flac'
#audio2raw(base_dir,meta_datas,finnish_speech_rootdirs,finnish_speech_extension,finnish_speech_outputfile_name,finnish_speech_outputlist_name)


finnish_dilact_outputfile_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/puhelahjat'
finnish_dilact_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/puhelahjat_meta'
finnish_dilact_rootdirs=["2020_part_14/audio_annotated"]
finnish_dilact_meta_data=["2020_part_14/puhelahjat_2020_part_14_meta.csv"]
finnish_speech_extension='.flac'
#audio2raw(base_dir,finnish_dilact_meta_data,finnish_dilact_rootdirs,finnish_speech_extension,finnish_dilact_outputfile_name,finnish_dilact_outputlist_name,is_split_on_silence=False)
temp_finnish_dilact_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/temp_puhelahjat.pkl'
temp1_finnish_dilact_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/temp1_puhelahjat.pkl'
#perform_train_test_split(finnish_dilact_outputfile_name,temp_finnish_dilact_outputlist_name,temp1_finnish_dilact_outputlist_name,seperator="_",position=0)
train_finnish_dilact_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/train_puhelahjat.pkl'
test_finnish_dilact_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_dilact/test_puhelahjat.pkl'
perform_train_test_split(temp1_finnish_dilact_outputlist_name,train_finnish_dilact_outputlist_name,test_finnish_dilact_outputlist_name,seperator="_",position=0)