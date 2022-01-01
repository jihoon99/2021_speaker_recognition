import pandas as pd
import numpy as np
import pickle

import os
import librosa
import librosa.display
import sklearn.preprocessing
import warnings
from tqdm import tqdm

import torch
import torch_optimizer as custom_optim
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn

import torchaudio

from pytorch_metric_learning import losses, miners, reducers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.utils import common_functions as c_f
from distances import CosineSimilarity

from sklearn.metrics import f1_score
import math
import argparse
import nsml
from nsml import DATASET_PATH

from patches import ConvMixer
from preprocess import make_left_right
from resnet import resnet50, resnet18, resnet34, resnet101, resnet34triplet, resnet34Contrastive, ResNetSE_angler, ResNetSE_concat, ResNetSEContrastive, ResNetSE_arcloss, ResNet_Trans
from Custom import CustomDataset, data_split, CustomDataset_2output, FinalDataset2output, FinalDatasetTriple, FinalDatasetTriple_infer
from cnnlstm import SpeechRecognitionModel, SpeechRecognitionModelShamCosine
from sklearn import preprocessing

import torch
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# kwargs = {'num_workers': 4, 'pin_memory': True}
kwargs = {'pin_memory': True}


def models(config, model_n = None):
    if config.version == 1:
        model = SpeechRecognitionModel(
            n_cnn_layers=3,
            n_rnn_layers=5,
            rnn_dim=512,
            n_class=1,  # 1 when BCEloss, 2 when
            n_feats=128,
            stride=2,
            dropout=0.1
        ).to(device)

    if config.version == 2 or model_n is not None:
        if model_n == 'resnet':
            model = resnet34().to(device)
            return model
        elif model_n == 'trans':
            model = resnet34().to(device)
            return model

    if config.version == 3:
        model = resnet34triplet().to(device)
        return model

    if config.version == 4:
        # cnnmixup
        model = ConvMixer(128, 34).to(device)
        return model
    
    if config.version == 5:
        return resnet34Contrastive().to(device)

    if config.version == 7:
        # output : [bs, 50]
        print('model : thin resnet SEblock with concat')
        return ResNetSE_concat().to(device)

    if config.version == 9:
        print("model : resnet SEblock with contrasive loss")
        return ResNetSEContrastive().to(device)
    
    if config.version == 10:
        print(config.version)
        print("model : thin SEResnet34")
        return ResNetSE_arcloss().to(device)


## hop_length has changed from 160 to 100
## mfcc


def wav2image_tensor(path, config, sr=16000, n_mfcc=128, n_fft=400, hop_length=100, max_len=1000):
    '''
    여기 바꾸면 custom.wave2image도 바꿔야함.
    
    '''
    if not config.mel: # if not False
        audio, _ = soundfile.read(path)
        audio, _ = librosa.effects.trim(audio)
        mfcc = librosa.feature.mfcc(
            audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        def pad2d(a, i): return a[:, 0:i] if a.shape[1] > i else np.hstack(
            (a, np.zeros((a.shape[0], i-a.shape[1]))))
        padded_mfcc = pad2d(mfcc, max_len).reshape(
            1, n_mfcc, max_len)  # 채널 추가
        padded_mfcc = torch.tensor(padded_mfcc, dtype=torch.float)

        return padded_mfcc

    if config.mel:
        '''
        여기서 멜 변환을 해줫으니, 모델에서 할필요없음. // x = self.instancenorm(x).unsqueeze(1).detach()만 해주면됨.
        '''
        audio = loadWAV(path, max_frames = 1200)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=config.n_mels)
                audio = torch.FloatTensor(audio)
                x = torchfb(audio) + 1e-6
                x = x.log()
                x = x.squeeze(0).detach() #[1,40,202] -> [40,202]
                # x = x.detach()
        return x  #[40, 202]


def loadWAV(filename, max_frames=600, evalmode=False, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audio, _ = librosa.effects.trim(audio)

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
    


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]



###### =================================================================================================##############








def loaders(POC, config, batch_size, n_sample, train_df):
    if POC:
        if config.version != 3:

            result, df = make_left_right(train_df, iteration = 2)  # iter will be 8

            tmp_result = result.iloc[:n_sample]  # 100개 샘플링 : df
            # instance들을 하나로 묶어.(POC모드라서 있는것,,)
            tmp_list = tmp_result['left_path'].values.tolist() + tmp_result['right_path'].values.tolist()
            df = df.loc[tmp_list]  # 해당하는 데이터만 가져와

            # mfcc를 한번에 해줄것임.
            mfcc_source = dict()  # {1:mfcc, 2:mfcc, ...}
            for idx, row in df.iterrows():
                try:
                    mfcc_source[idx]
                except:
                    mfcc_source[idx] = wav2image_tensor(row.file_name, config)

            # 확인용 print
            print(mfcc_source[tmp_list[0]].shape)

            # train valid split
            train_index, valid_index = data_split(tmp_result)
            valid_df = tmp_result.loc[valid_index].reset_index(drop=True)
            train_df = tmp_result.loc[train_index].reset_index(drop=True)

            # 확인용 print
            print(valid_df.head())
            print(train_df.head())

            # train과 source가 같이 들어갑니다.

            # 
            train_dataset = FinalDataset2output(left_path=train_df['left_path'],
                                                right_path=train_df['right_path'],
                                                label=train_df['label'],
                                                source=mfcc_source,
                                                mode='train',
                                                config=config)

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)

            valid_dataset = FinalDataset2output(left_path=valid_df['left_path'],
                                                right_path=valid_df['right_path'],
                                                label=valid_df['label'],
                                                source=mfcc_source,
                                                mode='valid',
                                                config=config)

            valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)
            print(f'{next(iter(train_dataset))}')

            return train_dataloader, valid_dataloader, mfcc_source
    
        # POC = True & version ==3 (POC용 tripletmarginloss) 여기만 수정함! (11.07. 혜인)
        if config.version == 3:
            print('loader just begun')
            result, df = make_left_right(train_df, 2)  # iter will be 8
            tmp_result = result.iloc[:n_sample]  # 100개 샘플링 : df

            # instance들을 하나로 묶어.(POC모드라서 있는것,,)
            tmp_list = tmp_result['left_path'].values.tolist() + tmp_result['right_path'].values.tolist()
            df = df.loc[tmp_list]  # 해당하는 데이터만 가져와

            # mfcc를 한번에 해줄것임.
            mfcc_source = dict()  # {1:mfcc, 2:mfcc, ...}
            for idx, row in df.iterrows():
                try:
                    mfcc_source[idx]
                except:
                    mfcc_source[idx] = wav2image_tensor(row.file_name, config)



            # 확인용 Print
            print(mfcc_source[tmp_list[0]].shape)

            # train valid split
            train_index, valid_index = data_split(tmp_result)
            valid_df = tmp_result.loc[valid_index].reset_index(drop=True)
            train_df = tmp_result.loc[train_index].reset_index(drop=True)
            print(valid_df.head())
            print(train_df.head())

            # speaker label encoding
            le = preprocessing.LabelEncoder()
            le.fit(df['speaker'])
            df['speaker'] = le.transform(df['speaker'])

            # 여기까지는 위와 동일.
            # ------------------------------------<Triplet Loss>--------------------------------------------
            
            #   train_dataset_similarity : (mfcc, speaker_id)  -> embedding 학습
            #   train_dataset : (left, right, label)
            #   valid_dataset : (left, right, label) -
                  
            # / mode = False로 하여 dropout을 안함. / config.BCL이 False로 되어있어야함.

            # <train_dataset> {'X' : padded_mfcc, 'Y': label(speaker_id)}
            train_dataset_similarity = FinalDatasetTriple(
                                                df = df,  # df는 376번에서 가져왓음. 372번이라고 봐도됨.
                                                source = mfcc_source,
                                                mode = 'train',
                                                config = config)
            
            train_dataloader_similarity = torch.utils.data.DataLoader(dataset = train_dataset_similarity,
                                                                        batch_size = batch_size,
                                                                        shuffle = True,
                                                                        **kwargs)
            print(next(iter(train_dataloader_similarity)))

            # train_dataloader, valid_dataloader의 아웃풋은 X_1, X_2, label이다.
            train_dataset = FinalDatasetTriple_infer(left_path = tmp_result['left_path'],
                                                    right_path = tmp_result['right_path'],
                                                    label = tmp_result['label'],
                                                    source=mfcc_source,
                                                    mode='train',
                                                    df = df,
                                                    config = config)

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)


            # print(f'similarity : {next(iter(train_dataloader_similarity))}')
            # print(f'train_dataset : {next(iter(train_dataloader))}')
            return train_dataloader_similarity, train_dataloader, mfcc_source



    # POC가 아닐경우
    else:
        if config.version != 3: # triple loss가 아닐경우.
            result, df = make_left_right(train_df, iteration=config.iteration)
            mfcc_source = dict()
            for idx, row in tqdm(df.iterrows()):
                try:
                    mfcc_source[idx]
                except:
                    #------------------------------------ max_len 바꿔서도 해보기 ########################################
                    mfcc_source[idx] = wav2image_tensor(row.file_name, config)

            print(mfcc_source[0].shape)

            train_index, valid_index = data_split(result)
            valid_df = result.loc[valid_index].reset_index(drop=True)
            train_df = result.loc[train_index].reset_index(drop=True)
            print(valid_df.head())
            print(train_df.head())


            train_dataset = FinalDataset2output(left_path=train_df['left_path'],
                                                right_path=train_df['right_path'],
                                                label=train_df['label'],
                                                source=mfcc_source,
                                                mode='train',
                                                config=config)

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)

            valid_dataset = FinalDataset2output(left_path=valid_df['left_path'],
                                                right_path=valid_df['right_path'],
                                                label=valid_df['label'],
                                                source=mfcc_source,
                                                mode='valid',
                                                config=config)

            valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)

            print(next(iter(train_dataset)))

            return train_dataloader, valid_dataloader, mfcc_source

        if config.version == 3:
            result, df = make_left_right(train_df, iteration=2)
            mfcc_source = dict()
            for idx, row in tqdm(df.iterrows()):
                try:
                    mfcc_source[idx]
                except:
                    #------------------------------------ max_len 바꿔서도 해보기 ########################################
                    mfcc_source[idx] = wav2image_tensor(row.file_name, config, max_len = 1000)

            print(mfcc_source[0].shape)

            train_index, valid_index = data_split(result)
            valid_df = result.loc[valid_index].reset_index(drop=True)
            train_df = result.loc[train_index].reset_index(drop=True)
            print(valid_df.head())
            print(train_df.head())

            # speaker label encoding
            le = preprocessing.LabelEncoder()
            le.fit(df['speaker'])
            df['speaker'] = le.transform(df['speaker'])



            # forArcTrainDf = []
            # for idx in range(df):
            #     forArcTrainDf.append({"speaker":df.loc[idx]['speaker'], "x" : mfcc_source[idx]})




            # forArcTrainDf = pd.DataFrame(forArcTrainDf)

            # forArcAgument = []
            # d_out = nn.Sequential(
                # torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                # torchaudio.transforms.TimeMasking(time_mask_param=100)
            # )

            # for idx in range(df):
                



            # 여기까지는 위와 동일.

            # train_datasetd은 60,000개에 대해서 할것임. / mode = False로 하여 dropout을 안함. / config.BCL이 False로 되어있어야함.
            # df : filename, speaker, pick (label encoding된 speaker)
           # <train_dataset> {'X' : padded_mfcc, 'Y': label(speaker_id)}
            train_dataset_similarity = FinalDatasetTriple(
                                                df = df,  # df는 376번에서 가져왓음. 372번이라고 봐도됨.
                                                source = mfcc_source,
                                                mode = 'False',
                                                config = config)
            
            train_dataloader_similarity = torch.utils.data.DataLoader(dataset = train_dataset_similarity,
                                                                        batch_size = batch_size,
                                                                        shuffle = True,
                                                                        **kwargs)

            print(next(iter(train_dataloader_similarity)))

            #train_dataloader, valid_dataloader의 아웃풋은 X_1, X_2, label이다.
            train_dataset = FinalDatasetTriple_infer(left_path = train_df['left_path'],
                                                    right_path = train_df['right_path'],
                                                    label = train_df['label'],
                                                    source=mfcc_source,
                                                    mode='train',
                                                    df = df,
                                                    config = config)

            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        **kwargs)

            return train_dataloader_similarity, train_dataloader, mfcc_source

def get_all_embeddings(dataset, model):
    # dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(data_device= device)
    return tester.get_all_embeddings(dataset, model)

def predict(x1, x2, model):
    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)
    with torch.no_grad():
        dist = model(x1, x2).cpu().numpy()
        return dist.flatten()


def training(config, model, optimizer, criterion, scaler, total_batch, epoch, l_norm=None, mining_funcs = None, train_dataloader = None, source = None, margin = None):
    avg_cost = 0
    avg_acc = []
    avg_label = []

    if config.version == 1 or config.version == 2 or config.version == 4 or config.version == 6 or config.version == 7:
        model.train()
        for idx, batch in tqdm(enumerate(train_dataloader)):
            
            
            if config.version == 1:
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    hypothesis = model(X)
                    cost = criterion(hypothesis, Y)
                    cost += l_norm(model, l_norm='L1')
                scaler.scale(cost).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()

                avg_cost += cost / total_batch
                tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()
                avg_acc += tmp.tolist()
                avg_label += Y.detach().cpu().numpy().tolist()

            
            # 모델이 두개의 인풋을 받아 들일때,
            elif config.version == 2 or config.version == 4 or config.version == 7:
                Y = batch['Y'].to(device).view(-1)  # |Y| = 64
                X_1 = batch['X_1'].to(device)
                X_2 = batch['X_2'].to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    hypothesis = model(X_1, X_2)
                    cost = criterion(hypothesis, Y)

                scaler.scale(cost).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()

                avg_cost += cost / total_batch
                tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()

                avg_acc += tmp.tolist()
                avg_label += Y.detach().cpu().numpy().tolist()
                if idx % 1000 == 0:
                    print(
                        f'[Epoch {epoch} Itereration {idx}] : Loss = {float(avg_cost):>.9}, f1 = {f1_score(avg_label, avg_acc)}')

        return avg_cost, avg_acc, avg_label, model
    
    # tripletloss 수정(11.07)
    if config.version == 3:

        model.train()


        for idx, batch in tqdm(enumerate(train_dataloader)):
            X = batch['X'].to(device) # padded_mfcc ([300, 1, 128, 1000])
            Y = batch['Y'].to(device) # label(speaker_id) ([300])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                hypothesis = model(X) # embedding값 (300, 400)
                
                # tripletloss : 갈수록 어려운 sample(margin 가까이 있는) 학습!
                if epoch < 100 :
                    mining_func = mining_funcs["mining_func_semihard"]
                else :
                    mining_func = mining_funcs["mining_func_hard"]

                # indices_tuple: (ancor_idx, positive_idx, negative_idx) -> len은 463033, 14918 등 그때그때 다름
                indices_tuple = mining_func(hypothesis, Y)
                cost = criterion(hypothesis, Y, indices_tuple)

            scaler.scale(cost).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            config.max_grad_norm,)
            scaler.step(optimizer)
            scaler.update()

            avg_cost += cost / total_batch
        
        return model, avg_cost


    # 모델이 두개의 인풋을 받아 들이는데, threshold가 필요할때,
    if config.version == 5 or config.version == 9:
        model.train()
        for idx, batch in tqdm(enumerate(train_dataloader)):
            Y = batch['Y'].to(device).view(-1)  # |Y| = 64
            X_1 = batch['X_1'].to(device)
            X_2 = batch['X_2'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                hypothesis = model(X_1, X_2) # hypothesis == distance
                cost = criterion(hypothesis, Y)

            scaler.scale(cost).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            avg_cost += cost / total_batch
            # tmp = torch.round(torch.sigmoid(
            #     hypothesis)).detach().cpu().numpy()
            # avg_acc += tmp.tolist()
            # avg_label += Y.detach().cpu().numpy().tolist()
            if idx % 500 == 0:
                print(f'[Epoch {epoch} Itereration {idx}] : Loss = {float(avg_cost):>.9}')
        print(avg_cost)
        return avg_cost, model


    if config.version == 10:
        model.train()
        for idx, batch in tqdm(enumerate(train_dataloader)):
            Y = batch['Y'].long().to(device).view(-1)
            X_1 = batch['X_1'].to(device)
            X_2 = batch['X_2'].to(device)

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            raw_logits = model(X_1, X_2)
            raw_logits = raw_logits[:,:128] # 두개 결합한거.
            output = margin(raw_logits, Y)
            total_loss = criterion(output, Y)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

            avg_cost += total_loss / total_batch

            if idx % 500 == 0:
                print(f'[Epoch {epoch} Itereration {idx}] : Loss = {float(avg_cost):>.9}')
        print(avg_cost)
        return avg_cost, model

    

# [TripletLoss]
# <train_dataset> {'X' : padded_mfcc, 'Y': label(speaker_id)} 추가 
def validating(config, valid_dataloader, model, criterion, total_batch, epoch, \
                avg_cost, avg_acc=None, avg_label=None, source = None, train_datasets = None, batch_size = None, mode = 'train', best_th = None):
    if config.version == 1 or config.version == 2 or config.version == 4 or config.version == 7:
        val_cost = 0
        val_acc = []
        val_label = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):

                if config.version == 1:
                    X = batch['X'].to(device)
                    Y = batch['Y'].to(device)

                    with torch.cuda.amp.autocast():
                        hypothesis = model(X)
                        cost = criterion(hypothesis, Y)

                    val_cost += cost/total_batch
                    tmp = torch.round(torch.sigmoid(
                        hypothesis)).detach().cpu().numpy()
                    val_acc += tmp.tolist()
                    val_label += Y.detach().cpu().numpy().tolist()

                elif config.version == 2 or config.version == 4 or config.version == 7:
                    Y = batch['Y'].to(device)
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2)
                        cost = criterion(hypothesis, Y)
                    val_cost += cost/total_batch
                    tmp = torch.round(torch.sigmoid(hypothesis)).detach().cpu().numpy()

                    val_acc += tmp.tolist()
                    val_label += Y.detach().cpu().numpy().tolist()

            print(tmp[:15])
            print(Y[:15])
            print(f'[Epoch: {epoch + 1:>4}] cost = {float(avg_cost):>.9} val cost = {float(val_cost):>.9} f1 = {f1_score(avg_label, avg_acc)} val f1 = {f1_score(val_label, val_acc)}')

        return model, config

    # Tripletloss (11.07 추가)
    if config.version == 3:
        model.eval()
        acc_by_threshold = {'thr_0.6':0, 'thr_0.7': 0, 'thr_0.8': 0,'thr_0.9': 0}

        with torch.no_grad():
            for batch in tqdm(valid_dataloader):

                Y = batch['Y'].to(device)
                X_1 = batch['left'].to(device)
                X_2 = batch['right'].to(device)
                
                # threshold 에 따라 inference 실행
                for threshold_key in acc_by_threshold.keys():
                    threshold_num = float(threshold_key.split("_")[1])
                    match_finder = MatchFinder(distance=CosineSimilarity(), threshold= threshold_num)
                    inference_model = InferenceModel(model, match_finder=match_finder)
                    acc_by_threshold[threshold_key] += round(sum(inference_model.is_match(X_1, X_2)==Y.detach().to('cpu').numpy())/(len(Y)*total_batch),3)

                val_acc = round(sum(acc_by_threshold.values())/len(acc_by_threshold.keys()),3)
                
            print(f'acc_by_threshold: {acc_by_threshold}')
            print(f'val_acc : {val_acc} ')
 
        best_idx = np.argmax(np.array(list(acc_by_threshold.values())))
        best_thresold = float(str(list(acc_by_threshold.keys())[best_idx]).split("_")[1])
        print(f"best threshold: {list(acc_by_threshold.keys())[best_idx]} ({list(acc_by_threshold.values())[best_idx]}) ")
        return model, config, best_thresold

    if config.version == 5 or config.version == 9:
        if mode == 'train':
            '''
            what i get as input 
            avg_cost, model, criterion, total_batch, epochs, source, 
            '''
            val_cost = 0
            val_label = []
            pos_distance = []
            neg_distance = []
            distances = []
            ys = []

            model.eval()
            with torch.no_grad():
                for batch in tqdm(valid_dataloader):
                    # load data
                    Y = batch['Y'].to(device)
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2)
                        cost = criterion(hypothesis, Y)
                        distance = predict(model, X_1, X_2) # distance : numpy

                    val_cost += cost/total_batch
                    # print(f'{list(Y.detach().cpu().numpy()==1)[:10]}')
                    # print(distance[:10])
                    pos_distance += distance[Y.detach().cpu().numpy() == 1].tolist() # list
                    # print(distance[Y.detach().cpu().numpy() == 1].tolist()[:10])
                    neg_distance += distance[Y.detach().cpu().numpy() != 1].tolist() # list
                    distances += distance.tolist() # list
                    ys += Y.detach().cpu().numpy().tolist()

                # 완성된 pos_dis, neg_dis에서 계산해야됨.
                distances = np.array(distances)
                a_threshold, wa_threshold = calculate_threshold(pos_distance, neg_distance)
                val_avg_threshold_acc = calculate_acc(distances, y = ys, thr = a_threshold)
                val_wavg_threshold_acc = calculate_acc(distances, y = ys, thr = wa_threshold)

            ys = np.array(ys)
            
            print(f'val_predict : {val_avg_threshold_acc[:10]}')
            print(f'val_ground_truth : {ys[:10]}')
            # print(f'{np.array(val_avg_threshold_acc)[:10]}')
            # print(f'{ys[:10]}')
            print(f'[Epoch: {epoch:>4}] cost = {float(avg_cost):>.9} val cost = {float(val_cost):>.9} a_acc = {np.mean(val_avg_threshold_acc == ys)} wa_acc = {np.mean(val_wavg_threshold_acc == ys)}')
            print(f'pos_dis : {np.mean(pos_distance)} neg_dis : {np.mean(neg_distance)} a_thr = {float(a_threshold):>9} wa_thr = {float(wa_threshold):>9}')
            print(f'margin : {np.mean(neg_distance)-np.mean(pos_distance)}')
            best_threshold = calculate_best_th(val_avg_threshold_acc, val_wavg_threshold_acc, a_threshold, wa_threshold)
            return model, config, best_threshold


        if mode == 'test':

            infer_answer = []
            distances = []

            model.eval()
            with torch.no_grad():
                for batch in tqdm(valid_dataloader):
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    with torch.cuda.amp.autocast():
                        hypothesis = model(X_1, X_2)
                        distance = predict(model, X_1, X_2)
                    
                    distances += distance.tolist()

                distances = np.array(distances)
                answer = calculate_acc(distances, thr = best_th, mode = 'infer')
                print(answer)
            return answer.tolist()

    if config.version == 10:
        if mode == 'train':

            model.eval()
            with torch.no_grad():
                # lfs = np.array([[]])
                # rts = np.array([[]])
                # ys = np.array([[]])
                cnt = 0
                for batch in tqdm(valid_dataloader):
                    Y = batch['Y'].to(device)
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    with torch.cuda.amp.autocast():
                        res = model(X_1, X_2)
                    res = res[:,128:]
                    lf = res[:,:128].detach().cpu().numpy()
                    rt = res[:,128:].detach().cpu().numpy()
                    Y = Y.detach().cpu().numpy()
                    if cnt == 0:
                        lfs = lf
                        rts = rt
                        ys = Y
                    else:
                        lfs = np.concatenate((lfs, lf), 0)
                        rts = np.concatenate((rts, rt), 0)
                        ys = np.concatenate((ys, Y), 0)

                    cnt += 1

                threshold, acc, pos_d, neg_d = evaluate_fold(lfs, rts, ys)
                answer = calculate_ys(lfs, rts, threshold)

                print(f'val_predict : {answer[:10]}')
                print(f'val_ground_truth : {ys[:10]}')
                print(f'[Epoch: {epoch:>4}] cost = {float(avg_cost):>.9} a_acc = {acc}')
                print(f'threshold : {threshold}, pos dis : {pos_d}, neg dis{neg_d}')

            return model, config, threshold

        if mode == 'test':

            model.evel()
            with torch.no_grad():
                # lfs = np.array([])
                # rts = np.array([])
                # ys = np.array([])
                cnt = 0
                for batch in tqdm(valid_dataloader):
                    Y = batch['Y'].to(device)
                    X_1 = batch['X_1'].to(device)
                    X_2 = batch['X_2'].to(device)
                    # with torch.cuda.amp.autocast():
                    res = model(X_1, X_2)
                    res = res[:,128:]
                    lf = res[:,:128].detach().cpu().numpy()
                    rt = res[:,128:].detach().cpu().numpy()
                    Y = Y.detach().cpu().numpy()

                    if cnt == 0:
                        lfs = lf
                        rts = rt
                        ys = Y
                    else:
                        lfs = np.concatenate((lfs, lf), 0)
                        rts = np.concatenate((rts, rt), 0)
                        ys = np.concatenate((ys, Y), 0)

                answer = calculate_ys(lfs, rts, best_th)
            return answer.tolist()


def calculate_ys(lf, rt, threshold):

    mu = np.sum(np.mean(np.concatenate((lf, rt),0)), 0)
    mu = np.expand_dims(mu, 0)

    featureLs = lf - mu
    featureRs = rt - mu
    featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
    featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

    scores = np.sum(np.multiply(featureLs, featureRs), 1)
    for idx,score in enumerate(scores):
        if score > threshold:
            scores[idx] = 1
        else:
            scores[idx] = 0

    return scores

def evaluate_fold(lf, rt, ys):
    mu = np.sum(np.mean(np.concatenate((lf, rt),0)), 0)
    mu = np.expand_dims(mu, 0)

    featureLs = lf - mu
    featureRs = rt - mu
    featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
    featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

    scores = np.sum(np.multiply(featureLs, featureRs), 1)
    threshold = getThreshold(scores, ys, 10000)
    acc, pd, nd = getAccuracy(scores, ys, threshold)
    return threshold, acc, pd, nd


def getThreshold(scores, ys, thrNum):
    '''
    at version 10, validation
    '''
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i],_,_ = getAccuracy(scores, ys, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def getAccuracy(scores, flags, threshold):
    '''
    at version 10, validation
    '''
    p = np.sum(scores[flags == 1] > threshold)
    pd = np.mean(scores[flags == 1])
    n = np.sum(scores[flags == 0] < threshold)
    nd = np.mean(scores[flags == 0])
    return 1.0 * (p + n) / len(scores), pd, nd



# Convenience function to make predictions
def predict(model, x1, x2):
    device = x1.device

    with torch.no_grad():
        dist = model(x1, x2).cpu().numpy()
        return dist.flatten()


def calculate_threshold(pos, neg):
    len_pos = len(pos)
    len_neg = len(neg)
    len_total = len_pos + len_neg
    w_avg = (len_pos/len_total)*np.mean(pos) + (len_neg/len_total)*np.mean(neg)
    avg = (np.mean(pos) + np.mean(neg))/2

    return avg, w_avg

def calculate_acc(distance, thr, y=None, mode='train'):
    # distances : list
    # thr : float
    # y : tensor
    thr = float(thr)
    distance = np.array(distance)

    if mode == 'train':
        y = np.array(y)
        # print(f'distance : {distance[:10]}')
        # print(f'threshold : {thr}')
        # print(f'inside calculate_acc y looks like:{y[:10]}')
        result = []
        for i in distance:
            if i < thr:
                result.append(1)
            else:
                result.append(0)
        result = np.array(result)
        # print(f'distance after booling : {distance[:10]}')
        # print(f'distance result : {distance[y==1][:10]}')
        return result # array를 내뱉고 계산은 밖에서 햇네

    else:
        result = []
        for i in distance:
            if i < thr:
                result.append(1)
            else:
                result.append(0)
        result = np.array(result)

        return result

'''
Traceback (most recent call last):
  File "main.py", line 478, in <module>
    model, config, best_threshold = validating(config=config,
  File "/app/versions.py", line 571, in validating
    val_avg_threshold_acc = calculate_acc(distances, y = Y, thr = a_threshold)
  File "/app/versions.py", line 632, in calculate_acc
    return list(distance == y)
TypeError: 'bool' object is not iterable
User session exited

'''


def calculate_best_th(a, wa, a_threshold, wa_threshold):
    aa = sum(a)/len(a)
    wawa = sum(wa)/len(wa)

    if aa > wawa:
        return a_threshold
    else:
        return wa_threshold



