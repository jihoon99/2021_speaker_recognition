import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import sklearn
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchaudio

import numpy
import random
import os
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

def loadWAV(filename, max_frames=200, evalmode=False, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat # (1, 32240) // eval mode에서는 (10, 32240) 근데 왜 10배하는지 모르겟음.
    



class CustomDataset(Dataset):
    def __init__(self, left_path, right_path, max_len=1000, config=None, label=None, mode='train'):
        self.left_path = left_path
        self.right_path = right_path
        self.max_len = max_len
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

    def __len__(self):
        return len(self.left_path)

    def wav2image_tensor(self, path):
        audio, sr = librosa.load(path, sr=self.sr)
        audio, _ = librosa.effects.trim(audio)
        mfcc = librosa.feature.mfcc(
            audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, self.max_len).reshape(
            1, self.n_mfcc, self.max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.wav2image_tensor(left_path)
        right_padded_mfcc = self.wav2image_tensor(right_path)

        padded_mfcc = torch.cat([left_padded_mfcc, right_padded_mfcc], dim=0)

        if self.mode == 'train':
            label = self.label[i]

            if self.config.BCL:
                label = torch.tensor(label, dtype=torch.float)
            else:
                label = torch.tensor(label, dtype=torch.long)

            return {
                'X': padded_mfcc,
                'Y': label
            }
        else:
            return {
                'X': padded_mfcc
            }


class CustomDataset_2output(Dataset):
    def __init__(self, left_path, right_path, config=None, label=None, mode='train', max_len=1000):
        self.left_path = left_path
        self.right_path = right_path
        self.max_len = max_len
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

    def __len__(self):
        return len(self.left_path)

    def wav2image_tensor(self, path):
        if self.config.version == 1 or self.config.version == 2 or self.config.version == 3 or self.config.version == 4 or self.config.version == 5 or self.config.version == 6 or self.config.version == 8:
            audio, sr = librosa.load(path, sr=self.sr)
            audio, _ = librosa.effects.trim(audio)
            mfcc = librosa.feature.mfcc(
                audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
            def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
                (a, np.zeros((a.shape[0], i-a.shape[1]))))
            padded_mfcc = pad2d(mfcc, self.max_len).reshape(
                1, self.n_mfcc, self.max_len)  # 채널 추가
            padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

            return padded_mfcc  # [bs, 1, 128, 1000] or [bs, 1, 1000, 128]

        if self.config.version == 7 or self.config.version == 9 or self.config.version == 10:
            audio = loadWAV(path)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=40)
                    audio = torch.FloatTensor(audio)
                    x = torchfb(audio) + 1e-6
                    x = x.log()
                    x = x.squeeze(0).detach() #[1,40,202] -> [40,202]
                    # x = x.detach()
            return x  #[40, 202]

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.wav2image_tensor(left_path)
        right_padded_mfcc = self.wav2image_tensor(right_path)

        if self.mode == 'train':
            label = self.label[i]

            if self.config.BCL:
                label = torch.tensor(label, dtype=torch.float)
            else:
                label = torch.tensor(label, dtype=torch.long)

            return {
                'X_1': left_padded_mfcc,
                'X_2': right_padded_mfcc,
                'Y': label
            }
        else:
            return {
                'X_1': left_padded_mfcc,
                'X_2': right_padded_mfcc,
            }


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 100, 6000, 2)
        #    Conv     -> (?, 100, 6000, 32)
        #    Pool     -> (?, 50, 3000, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 50, 3000, 32)
        #    Conv      ->(?, 50, 3000, 64)
        #    Pool      ->(?, 25, 1500, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 25x200x64 inputs -> 2 outputs
        self.fc = torch.nn.Linear(25*1500*64, 2, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)  # fc 가중치 초기화

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten
        out = self.fc(out)
        return out


def df_hash_librosa(df, n_fft, hop_length):
    ''' this process is necessary for infering method
        {'path':librosa}
    '''
    # this way it takes 6 min
    print("df to hashing is on progress")
    left_ = list([i for i in df['left_paths'].drop_duplicates().values])
    right_ = list([i for i in df['right_paths'].drop_duplicates().values])

    folder_paths = left_ + right_
    folder_paths = list(set(folder_paths))

    hashing = dict()
    for path in tqdm.tqdm(folder_paths):
        print(path)
        x, _ = librosa.load(path)
        x, _ = librosa.effects.trim(x)

        hashing[path] = x
    print('hashing is done')
    return hashing



def data_split(dataframe, ratio=False):
    '''
    spliting dataset into train(0.9), test(0.1)
    input : dataframe
    return : [train_array, valid_array]
    '''
    # 200000 개가 -> 80기가 : mel-spectrum, stft // 84기가....
    if ratio:
        x = dataframe['right_path']
        y = dataframe['label']
        train, test, _, _ = train_test_split(
            x, y, test_size=ratio, random_state=28, stratify=y)
        dataframe = dataframe.loc[train.index]
    x = dataframe['right_path']
    y = dataframe['label']
    train, test, _, _ = train_test_split(
        x, y, test_size=0.15, random_state=28, stratify=y)
    return list(train.index), list(test.index)


class FinalDataset2output(Dataset):
    def __init__(self, left_path, right_path, source, config=None, label=None, mode='train'):
        self.left_path = left_path
        self.right_path = right_path
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label
            self.dropout1 = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )
        if self.mode == 'valid':
            self.label = label


    def __len__(self):
        return len(self.left_path)

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.source[left_path]
        right_padded_mfcc = self.source[right_path]

        if self.mode == 'train':
            left_padded_mfcc = self.dropout1(left_padded_mfcc)
            right_padded_mfcc = self.dropout1(right_padded_mfcc)

        label = self.label[i]

        if self.config.BCL:
            # BCL일때는 tensor type float을 요구함.
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return {
            'X_1': left_padded_mfcc,
            'X_2': right_padded_mfcc,
            'Y': label
        }

        # else:
        #     return {
        #         'X_1': left_padded_mfcc,
        #         'X_2': right_padded_mfcc,
        #     }



class FinalDatasetTriple(Dataset):
    def __init__(self, df, source, config=None, mode='train'):
        ''' df
        |    |  filename  |  speaker |  pick  |
        |  1 |  ...001.wav|    idx01 |        |
        |  2 |  ...002.wav|    idx02 |        |
                        ....

            2. filename에 해당하는 mfcc를 source로 부터 가져올것임.

            이 작업을 getitem에서 할 것임.

        return 은 {X, Y}이다.
        '''
        self.df = self.sampling(df).reset_index(drop = True)
        # self.df = df
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.dropout1 = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100)
            )

        print(f'length of final dataset : {len(self.df)}')

    def sampling(self, df, max_num = 200, min_num=100):
        picked_idx = []
        for speaker in df.speaker.unique():
            tmp = df[df.speaker == speaker]
            if len(tmp) > max_num:
                picked_idx += np.random.choice(tmp.index.tolist(), max_num, replace = False).tolist()
            elif 5*len(tmp) < min_num:
                numAdd = 100-len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), numAdd, replace = True).tolist()
            else: # 5배보다 작고 
                numAdd = max_num - len(tmp)
                picked_idx += np.random.choice(tmp.index.tolist(), numAdd, replace = True).tolist()
        
        return df.loc[picked_idx]

    def stretch_sound(self, data, sr=16000, rate=0.8):# stretch 해주는 것 테이프 늘어진 것처럼 들린다.
        stretch_data = librosa.effects.time_stretch(data, rate)
        return stretch_data

    def reverse_sound(self, data, sr=16000):# 거꾸로 재생
        data_len = len(data)
        data = np.array([data[len(data)-1-i] for i in range(len(data))])
        return data

    def wave2image_tensor(self,path, pre,sr=16000, n_mfcc=128, n_fft=400, hop_length=100, max_len=1000):
        audio, _ = soundfile.read(path)
        audio, _ = librosa.effects.trim(audio)
        # print(audio.shape)

        # if preprocess == 'No':
        #     pass
        # elif preprocess == 'reverse':
        #     audio = self.reverse_sound(audio)
        # else:
        #     audio = self.stretch_sound(audio, rate = preprocess)

        if pre == 'agument':
            augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ])

            audio = augment(samples=audio, sample_rate=sr)


        mfcc = librosa.feature.mfcc(
            audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, max_len).reshape(
            1, n_mfcc, max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        first = np.random.choice(['agument', 'no'], 1)[0]
        # if first == 'No':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'No')
        # elif first == 'stretch':
        #     preprocess = np.random.uniform(0.7, 1.3, 1)
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess=preprocess)
        # elif first == 'reverse':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'reverse')
        # elif first == 'zero':
        #     padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], preprocess = 'No')
        #     padded_mfcc = self.dropout1(padded_mfcc)
        padded_mfcc = self.wave2image_tensor(self.df.iloc[i]['file_name'], pre=first)




        # print(padded_mfcc.shape)

        # ith_idex = self.df.index[i]
            

        # padded_mfcc = self.source[ith_idex] # df의 i번째 filename에 해당하는 mfcc
        
        label = self.df['speaker'].iloc[i]
        label = torch.tensor(label, dtype=torch.long)

        return {
            'X': padded_mfcc,
            'Y': label
        }




class FinalDatasetTriple_infer(Dataset):
    def __init__(self, left_path, right_path, source, df, config=None, label=None, mode='train'):
        
        self.left_path = left_path
        self.right_path = right_path
        self.source = source
        self.max_len = 1000
        self.sr = 16000
        self.n_mfcc = 128
        self.n_fft = 400
        self.hop_length = 100
        self.mode = mode
        self.config = config

        if self.mode == 'train':
            self.label = label

            left_label = left_path.values
            right_label = right_path.values
            self.left_speaker_label = df.loc[left_label]['speaker'] # index = left_label
            self.right_speaker_label = df.loc[right_label]['speaker'] # index = right_label


    def __len__(self):
        return len(self.left_path)

    def __getitem__(self, i):
        left_path = self.left_path[i]
        right_path = self.right_path[i]

        left_padded_mfcc = self.source[left_path]
        right_padded_mfcc = self.source[right_path]


        if self.mode == 'train':
            label = self.label.iloc[i]
            l_label = self.left_speaker_label.iloc[i]
            r_label = self.right_speaker_label.iloc[i] 


        if self.config.BCL:
            # BCL일때는 tensor type float을 요구함.
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)


        if self.mode == 'train':
            return {
                'left': left_padded_mfcc,
                'right': right_padded_mfcc,
                'l_label' : l_label,
                'r_label': r_label,
                'Y': label
            }

        else: # inference 할때
            return {
                'left' : left_padded_mfcc,
                'right' : right_padded_mfcc,
            }

        # else:
        #     return {
        #         'X_1': left_padded_mfcc,
        #         'X_2': right_padded_mfcc,
        #     }




