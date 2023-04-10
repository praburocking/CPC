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
import librosa

from datetime import datetime









N_MELS=60
def audio2raw(base_dir,meta_datas,rootdirs,extension,outputfile_name,outputlist_name):


    file_names=[]
    class_names=[]
    chunk_length=15000
    total_file_not_found=0
    print(rootdirs)
    for i,rootdir in enumerate(rootdirs):
        if (i+1)<8:
            print("skipping ... "+str(i+1)+":: csv..."+meta_datas[i]+" :: dir..."+rootdir)
            continue
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
            file_name=row["recordingId"]+".flac"
            cur_dir=os.path.join(base_dir, rootdir)
            fullpath = os.path.join(cur_dir, file_name)
            if (os.path.isfile(fullpath)):
                flac = AudioSegment.from_file(fullpath, format='flac')
                #audio_chunks = split_on_silence(flac, min_silence_len=1000,silence_thresh=-16)
                audio_chunks = make_chunks(flac, chunk_length)
                for j, chunk in enumerate(audio_chunks):
                    output_name=row["recordingId"]+"_"+str(j)
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
            else:
                print(current_time+" :: file not found "+str(file_name),flush=True)
                total_file_not_found+=1
        output_file.to_csv(temp_output_csv_filename)
    print(total_file_not_found)
    
    h5f.close()

                    

finnish_speech_outputfile_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_15sec/puhelahjat'
finnish_speech_outputlist_name='/scratch/kcprmo/cpc/data/dataset/puhelahjat_15sec/puhelahjat_meta'
base_dir="/scratch/kcprmo/cpc/finnish_data/puhelahjat/v1/"
#finnish_speech_rootdirs=["2020_part_01/audio_annotated","2020_part_02/audio_annotated","2020_part_03/audio_annotated","2020_part_04/audio_annotated","2020_part_05/audio_annotated","2020_part_06/audio_annotated","2020_part_07/audio_annotated","2020_part_08/audio_annotated","2020_part_09/audio_annotated"]
#meta_datas=["2020_part_01/puhelahjat_2020_part_01_meta.csv","2020_part_02/puhelahjat_2020_part_02_meta.csv","2020_part_03/puhelahjat_2020_part_03_meta.csv","2020_part_04/puhelahjat_2020_part_04_meta.csv","2020_part_05/puhelahjat_2020_part_05_meta.csv","2020_part_06/puhelahjat_2020_part_06_meta.csv","2020_part_07/puhelahjat_2020_part_07_meta.csv","2020_part_08/puhelahjat_2020_part_08_meta.csv","2020_part_09/puhelahjat_2020_part_09_meta.csv"]
finnish_speech_rootdirs=["2020_part_01/audio_annotated","2020_part_02/audio_annotated","2020_part_03/audio_annotated","2020_part_04/audio_annotated","2020_part_05/audio_annotated","2020_part_06/audio_annotated","2020_part_07/audio_annotated","2020_part_08/audio_annotated","2020_part_09/audio_annotated"]
meta_datas=["2020_part_01/puhelahjat_2020_part_01_meta.csv","2020_part_02/puhelahjat_2020_part_02_meta.csv","2020_part_03/puhelahjat_2020_part_03_meta.csv","2020_part_04/puhelahjat_2020_part_04_meta.csv","2020_part_05/puhelahjat_2020_part_05_meta.csv","2020_part_06/puhelahjat_2020_part_06_meta.csv","2020_part_07/puhelahjat_2020_part_07_meta.csv","2020_part_08/puhelahjat_2020_part_08_meta.csv","2020_part_09/puhelahjat_2020_part_09_meta.csv"]
finnish_speech_rootdirs.extend(["2020_part_10/audio_annotated","2020_part_11/audio_annotated","2020_part_12/audio_annotated","2020_part_13/audio_annotated"])
meta_datas.extend(["2020_part_10/puhelahjat_2020_part_10_meta.csv","2020_part_11/puhelahjat_2020_part_11_meta.csv","2020_part_12/puhelahjat_2020_part_12_meta.csv","2020_part_13/puhelahjat_2020_part_13_meta.csv"])
finnish_speech_extension='.flac'
audio2raw(base_dir,meta_datas,finnish_speech_rootdirs,finnish_speech_extension,finnish_speech_outputfile_name,finnish_speech_outputlist_name)