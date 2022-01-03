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
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, reducers, testers
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
from resnet import resnet50, resnet18, resnet34, resnet101, ResNet_Trans
from Custom import CustomDataset, data_split, CustomDataset_2output, FinalDataset2output, FinalDatasetTriple, FinalDatasetTriple_infer
from cnnlstm import SpeechRecognitionModel, SpeechRecognitionModelShamCosine
from Loss.arcmargin import ArcMarginProduct
from versions import models, loaders, training, validating
from Loss import angleproto, aamsoftmax


print('torch version: ', torch.__version__)
warnings.filterwarnings(action='ignore')

# GPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ', device)

# 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
else:
    torch.manual_seed(42)




def full_path_df(root_path, df, mode='train'):
    '''
        convert path to nsml path
    '''

    # full_path_df = pd.DataFrame({'left_path': root_path + '/' + df['file_name'],
    #                              'right_path': root_path + '/' + df['file_name_']})
    df['file_name'] = root_path + '/' + df['file_name']
    return df


def full_path_df_infer(root_path, df):
    ''' 
        convert path to nsml path
        this is for inference
    '''
    full_path_df = pd.DataFrame({'left_path': root_path + '/' + df['file_name'],
                                 'right_path': root_path + '/' + df['file_name_']})

    return full_path_df


def max_len_check(df):
    '''
        To check length of wav
        For EDA
    '''
    def len_check(path, sr=16000, n_mfcc=100, n_fft=400, hop_length=160):
        audio, sr = librosa.load(path, sr=sr)
        mfcc = librosa.feature.mfcc(
            audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

        return mfcc.shape[1]

    left_len = df['left_path'].apply(lambda x: len_check(x))
    right_len = df['right_path'].apply(lambda x: len_check(x))

    left_max_len = left_len.max()
    right_max_len = right_len.max()

    return (max(left_max_len, right_max_len))



def save_checkpoint(checkpoint, dir):
    torch.save(checkpoint, os.path.join(dir))


def l_norm(model, l_norm='L1'):
    '''
        set L norms
    '''
    if l_norm == 'L1':
        L = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                L = L + torch.norm(param, 1)
        L = 10e-4 * L
    elif l_norm == 'L2':
        L = sum(p.pow(2.0).sum() for p in model.parameters())
    return L


def contrastive_loss(y, t):
    '''
        custom contrastive_loss
    '''
    nonmatch = F.relu(3 - y)  # max(margin - y, 0)
    return torch.mean(t * y**2 + (1 - t) * nonmatch**2)


def bind_model(model, parser):
    '''
        nsml wrapper : wrapping save, load, infer function
    '''

    def save(dir_name, *parser):
        '''
            save trained model to nsml system
        '''
        os.makedirs(dir_name, exist_ok=True)
        save_dir = os.path.join(dir_name, 'checkpoint')
        save_checkpoint(dict_for_infer, save_dir)

        with open(os.path.join(dir_name, "dict_for_infer"), "wb") as f:
            pickle.dump(dict_for_infer, f)

        print(f"학습 모델 저장 완료!")

    def load(dir_name, *parser):
        '''
            load saved model from nsml system
        '''
        save_dir = os.path.join(dir_name, 'checkpoint')

        global checkpoint
        checkpoint = torch.load(save_dir)

        model.load_state_dict(checkpoint['model'])

        global dict_for_infer
        with open(os.path.join(dir_name, "dict_for_infer"), 'rb') as f:
            dict_for_infer = pickle.load(f)

        print("    ***학습 모델 로딩 완료!")

    def infer(test_path, **kwparser):
        test_data = pd.read_csv(os.path.join(test_path, 'test_data', 'test_data'))
        root_path = os.path.join(test_path, 'test_data', 'wav')
        test_df = full_path_df_infer(root_path, test_data)

        kwargs = {'num_workers': 3, 'pin_memory': True}
        print(f'config.version : {config.version}')


        if config.version == 1:
            test_dataset = CustomDataset(
                test_df['left_path'], test_df['right_path'], max_len=1000, label=None, mode='test', config=config)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          **kwargs)
            model.eval()
            preds = []
            for batch in test_dataloader:
                X = batch['X'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = model(X)
                    pred = torch.tensor(torch.round(torch.sigmoid(
                        pred)), dtype=torch.long).cpu().numpy()
                    preds += list(pred)

        elif config.version == 2 or config.version == 4 or config.version == 7:
            test_dataset = CustomDataset_2output(test_df['left_path'], test_df['right_path'], label=None, mode='test', config=config)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          **kwargs)
            model.eval()
            preds = []
            for batch in test_dataloader:
                X_1 = batch['X_1'].to(device)
                X_2 = batch['X_2'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = model(X_1, X_2)
                    pred = torch.tensor(torch.round(torch.sigmoid(
                        pred)), dtype=torch.long).detach().cpu().numpy()
                    preds += list(pred)


        # tripletloss (11.08 수정)
        elif config.version == 3:
            test_dataset = CustomDataset_2output(test_df['left_path'],
                                                 test_df['right_path'],
                                                 label=None,
                                                 mode='test',
                                                 config = config,
                                                 max_len = 1000)

            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          **kwargs)

            model.eval()
            preds = []
            best_threshold = dict_for_infer['best_threshold']

            print(f"(infer) best_threshold : {best_threshold}")
            match_finder = MatchFinder(
                distance=CosineSimilarity(), threshold=best_threshold)
            inference_model = InferenceModel(model, match_finder=match_finder)

            for batch in test_dataloader:
                X_1 = batch['X_1'].to(device)
                X_2 = batch['X_2'].to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        pred = inference_model.is_match(X_1, X_2).astype(
                            np.int64)  # True/False -> int로
                    preds += list(pred)


        elif config.version == 5 or config.version == 9 or config.version == 10:
            test_dataset = CustomDataset_2output(test_df['left_path'], test_df['right_path'], label=None, mode='test', config = config)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          **kwargs)
            preds = validating(config=config,
                                valid_dataloader=test_dataloader,
                                model=model,
                                criterion=None,
                                total_batch=None,
                                epoch=None,
                                avg_cost=None,
                                mode = 'test',
                                best_th = dict_for_infer['best_threshold'])

            print(preds)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(prob, pred)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # 위의 포맷에서 prob은 리더보드 결과에 확률의 값은 영향을 미치지 않습니다(pred만 가져와서 채점).
        # pred에는 예측한 binary 혹은 1에 대한 확률값을 넣어주시면 됩니다.
        prob = [1]*len(preds)

        return list(zip(prob, preds))

    nsml.bind(save=save, load=load, infer=infer)


'''
mel spec - information
총 63,782 records / 847 speakers / 

count  847.000000
mean    75.303424
std     63.498767
min     14.000000
25%     31.500000
50%     55.000000
75%     96.000000
max    466.000000

99 :  1575
98 :  1425
97 :  1335
96 :  1271
95 :  1226
94 :  1210
93 :  1188
92 :  1156
91 :  1121
90 :  1082
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nia_test')
    parser.add_argument('--mode',      type=str, default='train', help = 'relate to nsml system. barely change')
    parser.add_argument('--epochs',    type=int, default=100,     help = 'number of epochs to train')
    parser.add_argument('--pause',     type=int, default=0,       help = 'relate to nsml system. barely change')
    parser.add_argument('--max_grad_norm',       default=5.,      help = 'maximum gradient for gradient clipping')
    # affects to custom.py/ label should be float -> 1
    parser.add_argument('--BCL',      default=False,    help = 'this is used when target is set to be binary. this should be True when version 1,2,4,7') # for version 2,4,7
    # 1:unimodal, 2:siamness, 3:tripleLoss  4,  5 : contrasive loss, 7: thin-resnet with anglerLoss, version
    # 9  : thin resnet with contrasive loss
    # 10 : thin resnet with arcface loss
    parser.add_argument('--version',  default=3,
        help = 'version 1 : unimodal with BCLoss and input two speakers\
                version 2 : siamness ResNet model with BCLoss or siamness transformer model with BCLoss and input two speakers\
                version 3 : ResNet with tripleLoss input one speaker\
                version 4 : ConvMixer with BCLoss and input two speaker - patches are all you need(ICLR 2021) \
                version 5 : ResNet with contrasive loss and input two speaker \
                version 7 : thin-SEResNet with anglerLoss and input two speaker \
                version 9 : thin-SEResNet with contrasive loss input two speaker \
                version 10 : this SEResNet pretrain with ArcMargin, get threshold with CrossEntropyLoss, and input one speaker ')

    parser.add_argument('--iteration', type=int, default = 2,       help = 'it works on version 5, 7, 9, 10')
    parser.add_argument('--mel',                 default = False,   help = 'it works on version 7, 9, 10') # True : version 7,9,10
    parser.add_argument('--n_mels',    type=int, default = 80,      help = 'it works on version 7, 9, 10') # for version 7, 9, 10
    parser.add_argument('--learning_rate',       default = 1e-4)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--POC_MODE',             default = False,  help = 'for testing since preprocess is time consuming work')
    config = parser.parse_args()

    learning_rate = config.learning_rate
    batch_size = config.batch_size

    scaler = torch.cuda.amp.GradScaler()
    model = models(config)
    bind_model(model=model, parser=config)  # related to nsml system


    print(config.version)


    if config.pause: # related to nsml system
        nsml.paused(scope=locals())


    if config.mode == 'train':

        train_path = os.path.join(DATASET_PATH, 'train/train_data')
        root_path = os.path.join(train_path, 'wav')
        train_label = pd.read_csv(os.path.join(train_path, 'train_info'))
        train_df = full_path_df(root_path, train_label, mode='test')

        '''
        데이터 프레임의 형태는 아래와 같은 모양이다.
        train_df : root/train/train_data/wav/idx00001, idx00002
        train_df

                | ind |                   file_name                         |   speaker   |
                ---------------------------------------------------------------------------
                |  0  |  /data/final_speaker/train/train_data/wav/idx00...  |  S000013_1  |
                |  1  |  /data/final_speaker/train/train_data/wav/idx00...  |  S000013_2  |
        '''

        print("train_df", train_df.head(3))
        print("train_label", train_label.head(3))

        # kwargs = {'num_workers': 3, 'pin_memory': False} # num_workers 옵션으로 인해 가끔 머신이 멈출때가 있다. 
        kwargs = {'pin_memory': True} 

        #### 전처리에 상당한 시간이 뺏기므로 테스트 용으로 실험해보는 ####
        POC_MODE = config.POC_Mode
        n_sample = 300
        if POC_MODE:
            # POC & not TripletLoss
            # -> left_padded_mfcc, right_padded_mfcc, label
            if config.version != 3:
                train_dataloader, valid_dataloader, mfcc_source = loaders(POC_MODE, 
                                                                            config, 
                                                                            batch_size, 
                                                                            n_sample, 
                                                                            train_df)

            # POC & TripletLoss
            #  혜인 (21.11.07)
            if config.version == 3:
                train_dataloader_similarity, train_dataloader, mfcc_source = loaders(POC_MODE, 
                                                                            config, 
                                                                            batch_size, 
                                                                            n_sample, 
                                                                            train_df)


        else: # 실제 트레이닝 할때
            # not POC & not TripletLoss
            if config.version != 3:
                train_dataloader, valid_dataloader, mfcc_source = loaders(POC_MODE, 
                                                                            config, 
                                                                            batch_size, 
                                                                            n_sample, 
                                                                            train_df)

            # not POC & TripletLoss
            #  혜인 (21.11.07)
            #   변경사항 : infer_dataloader, train_dataset 추가
            if config.version == 3:
                train_dataloader_similarity, train_dataloader,  mfcc_source = loaders(POC_MODE, 
                                                                            config, 
                                                                            batch_size, 
                                                                            n_sample, 
                                                                            train_df)



        ######################### criterion ################################################################

        if config.version == 1 or config.version == 2 or config.version == 4 or config.version == 7:
            # Loss : BCEWithLogitsLoss
            print(f'criterion : BCEWithLogitsLoss')
            criterion = torch.nn.BCEWithLogitsLoss().to(device)

        if config.version == 5 or config.version == 9:
            # Loss : Contrastive_loss
            criterion = contrastive_loss
            print('criterion : contrasiveLoss')

        if config.version == 8:
            # Loss : angleproto
            criterion = angleproto()

        if config.version == 3:
            # triple margin loss
            distance = CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            criterion = losses.TripletMarginLoss(margin=0.1,
                                                 distance=distance,
                                                 reducer=reducer,
                                                 triplets_per_anchor="all")  # margin 0.2 -> 0.05 수정(11.09)

            mining_func_easy = miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="easy")
            mining_func_semihard = miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="semihard")
            mining_func_hard = miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="hard")
            mining_funcs = {"mining_func_easy": mining_func_easy,
                            "mining_func_semihard": mining_func_semihard,
                            "mining_func_hard": mining_func_semihard}
            print("criterion : triple margin loss")

        if config.version == 10:
            # Loss : CrossEntropyLoss
            print('criterion : CrossEntropyLoss')
            criterion = torch.nn.CrossEntropyLoss().to(device)



        ########################## optimizer #################################################################
        if config.version == 10:
            margin = ArcMarginProduct(in_feature = 128, 
                                      out_feature = 2,
                                      m = 1.5).to(device)

            optimizer = custom_optim.RAdam([
                {'params':model.parameters()},
                {'params':margin.parameters()}], lr = learning_rate)

        else:
            optimizer = custom_optim.RAdam(model.parameters(), lr=learning_rate)




        ########################## testing the shape #######################################################
        total_batch = math.ceil(len(train_dataloader))

        print(f'학습 시작! : total_batch - {total_batch}')

        if config.version == 1:
            print('train_dataloader: ', next(
                iter(train_dataloader))['X'].shape)

        elif config.version == 2 or config.version == 4 or config.version == 7:
            print('train_dataloader: ', next(iter(train_dataloader))
                  ['X_1'].shape)  # [64, 1, 128, 1000]

        elif config.version == 3:
            print('train_dataloader: ', next(iter(train_dataloader))
                  ['left'].shape)  # [64, 1, 128, 1000]

        elif config.version == 5 or config.version == 9 or config.version == 10:
            print('train_dataloader: ', next(iter(train_dataloader))
                  ['X_1'].shape)  # [64, 1, 128, 1000]


        ########################## training start ###########################################################
        # 여기는 두개의 인풋을 넣는 모델이다.
        if config.version == 1 or config.version == 2 or config.version == 4 or config.version == 7:
            # epoch start
            for epoch in range(config.epochs):
                avg_cost, avg_acc, avg_label, model = training(config=config,
                                                               train_dataloader=train_dataloader,
                                                               model=model,
                                                               optimizer=optimizer,
                                                               criterion=criterion,
                                                               l_norm=l_norm,
                                                               scaler=scaler,
                                                               total_batch=total_batch,
                                                               epoch=epoch)

                model, config = validating(config=config,
                                           valid_dataloader=valid_dataloader,
                                           model=model,
                                           criterion=criterion,
                                           total_batch=total_batch,
                                           epoch=epoch,
                                           avg_cost=avg_cost,
                                           avg_acc=avg_acc,
                                           avg_label=avg_label)

                dict_for_infer = {
                    'model': model.state_dict(),
                    'config': config,
                }

                nsml.save(epoch) # nsml에 모델을 저장.



        if config.version == 5 or config.version == 9:
            for epoch in range(config.epochs):
                avg_cost, model = training(config=config,
                                            train_dataloader=train_dataloader,
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            l_norm=l_norm,
                                            scaler=scaler,
                                            total_batch=total_batch,
                                            epoch=epoch)

                model, config, best_threshold = validating(config=config,
                                                            valid_dataloader=valid_dataloader,
                                                            model=model,
                                                            criterion=criterion,
                                                            total_batch=total_batch,
                                                            epoch=epoch,
                                                            avg_cost=avg_cost)

                dict_for_infer = {
                    'model': model.state_dict(),
                    'config': config,
                    'best_threshold':best_threshold,
                }

                # if epoch % 2 == 1:
                nsml.save(epoch)

        if config.version == 10:
            for epoch in range(config.epochs):
                avg_cost, model = training(config=config,
                                            train_dataloader=train_dataloader,
                                            model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            l_norm=l_norm,
                                            scaler=scaler,
                                            total_batch=total_batch,
                                            epoch=epoch,
                                            margin = margin)

                model, config, best_threshold = validating(config=config,
                                                            valid_dataloader=valid_dataloader,
                                                            model=model,
                                                            criterion=criterion,
                                                            total_batch=total_batch,
                                                            epoch=epoch,
                                                            avg_cost=avg_cost)

                dict_for_infer = {
                    'model': model.state_dict(),
                    'config': config,
                    'best_threshold':best_threshold,
                }

                # if epoch % 2 == 1:
                nsml.save(epoch)




        # TripletLoss (11.07.혜인)
        if config.version == 3:
            total_batch_sim = math.ceil(len(train_dataloader_similarity))
            total_batch_val = math.ceil(len(train_dataloader))
            for epoch in range(config.epochs):
                model, avg_cost = training(config=config,
                                    model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    scaler=scaler,
                                    total_batch=total_batch_sim,
                                    epoch=epoch,
                                    train_dataloader=train_dataloader_similarity,
                                    mining_funcs=mining_funcs,
                                    source=mfcc_source)
                print("="*100)
                print(f"[Epoch: {epoch + 1:>4}] training 완료!")

                # test 시간이 오래걸려서 epoch 줄임 (30 배수에서 확인)
                if (epoch + 1) % 30 == 0:
                    print("---------test 시작! -> train_dataloader 수 : ",
                          len(train_dataloader))
                    model, config, best_threshold = validating(config=config,
                                                               valid_dataloader=train_dataloader,
                                                               model=model,
                                                               criterion=None,
                                                               total_batch=total_batch_val,
                                                               epoch=epoch,
                                                               avg_cost=avg_cost,
                                                               avg_acc=None,
                                                               avg_label=None,
                                                               source=mfcc_source
                                                               )
                    print(best_threshold)
                    dict_for_infer = {
                        'model': model.state_dict(),
                        'config': config,
                        'best_threshold': best_threshold
                    }

                    print(
                        f"---------test 완료! -> best_threshold = {best_threshold} ")
                    # if epoch % 2 == 1:
                    nsml.save(epoch)

    else:
        # kwargs = {'num_workers': 4, 'pin_memory': True}
        kwargs = {'pin_memory': True}
        device = torch.device("cuda:0")
        bind_model(parser=config)

        # bind_model(model = model, parser=config)
        if config.pause:
            nsml.paused(scope=locals())
