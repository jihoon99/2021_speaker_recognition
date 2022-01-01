import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# class CNNLayerNorm(nn.Module):
#     '''Layer normalization'''
#     def __init__(self, n_feats):
#         super(CNNLayerNorm, self).__init__()
#         self.layer_norm = nn.LayerNorm(n_feats)

#     def forward(self, x):
#         '''|x| = (batch, channel, feature, length)'''
#         # x = x.transpose(2, 3).contiguous() # (batch, channel, length, feature)
#         x = self.layer_norm(x)
#         return x.transpose(2, 3).contiguous() # (batch, channel, feature, length) 


# class ResidualCNN(nn.Module):
#     '''
#         Residual CNN
#     '''
#     def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
#         super(ResidualCNN, self).__init__()

#         self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
#         self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.layer_norm1 = CNNLayerNorm(n_feats)
#         self.layer_norm2 = CNNLayerNorm(n_feats)

#     def forward(self, x):
#         residual = x  # (batch, channel, feature, length)
#         x = self.layer_norm1(x) # bs, c, length, fe
#         x = F.gelu(x)
#         x = self.dropout1(x)
#         x = self.cnn1(x) # 
#         x = x.transpose(2,3).contiguous()
#         x = self.layer_norm2(x)
#         x = F.gelu(x)
#         x = self.dropout2(x)
#         x = self.cnn2(x)
#         x = x.transpose(2,3).contiguous()
#         x += residual
#         return x # (batch, channel, feature, length)???


# class BidirectionalGRU(nn.Module):
#     '''GRU Block'''
#     def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
#         super(BidirectionalGRU, self).__init__()

#         self.BiGRU = nn.GRU(
#             input_size=rnn_dim, hidden_size=hidden_size,
#             num_layers=1, batch_first=batch_first, bidirectional=True)
#         self.layer_norm = nn.LayerNorm(rnn_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.layer_norm(x)
#         x = F.gelu(x)
#         x, _ = self.BiGRU(x)
#         x = self.dropout(x)
#         return x


# class SpeechRecognitionModel(nn.Module):
    
#     def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
#         super(SpeechRecognitionModel, self).__init__()
#         n_feats = n_feats//2
#         self.cnn = nn.Conv2d(2, 32, 3, stride=stride, padding=3//2)  # [32, 2channel, 3,3]

#         self.rescnn_layers = nn.Sequential(*[
#             ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
#             for _ in range(n_cnn_layers)
#         ])
#         self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
#         self.birnn_layers = nn.Sequential(*[
#             BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
#                              hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
#             for i in range(n_rnn_layers)
#         ])

#         self.classifier = nn.Sequential(
#             nn.Linear(rnn_dim, rnn_dim),  
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(rnn_dim, n_class)
#         )

#         # torch.nn.init.xavier_uniform_(self.classifier.weight) # fc 가중치 초기화


#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.rescnn_layers(x)
#         print(x.shape)
#         sizes = x.size()
#         print(sizes)
#         x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
#         x = x.transpose(1, 2).contiguous() # (batch, length, feature)
#         print(x.shape)
#         x = self.fully_connected(x)
#         x = self.birnn_layers(x)
#         print(x.shape)
#         x = x.mean(dim=-2)
#         x = self.classifier(x)
#         return x


# # class SpeechRecognitionModel(nn.Module):

# #     def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
# #         super(SpeechRecognitionModel, self).__init__()
# #         n_feats = n_feats//2
# #         self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn으로 heirachal특성을 추출

# #         self.rescnn_layers = nn.Sequential(*[
# #             ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
# #             for _ in range(n_cnn_layers)
# #         ])
# #         self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
# #         self.birnn_layers = nn.Sequential(*[
# #             BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
# #                              hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
# #             for i in range(n_rnn_layers)
# #         ])
# #         self.classifier = nn.Sequential(
# #             nn.Linear(rnn_dim*2, rnn_dim),  
# #             nn.GELU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(rnn_dim, rnn_dim)
# #         )

# #     def forward_once(self, x):
# #         x = self.cnn(x)
# #         x = self.rescnn_layers(x)
# #         sizes = x.size()
# #         x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
# #         x = x.transpose(1, 2) # (batch, length, feature)
# #         x = self.fully_connected(x)
# #         x = self.birnn_layers(x)
# #         x = x.mean(dim=-2)
# #         x = self.classifier(x)
# #         return x

# #     def forward(self, x, y):
# #         output1 = self.forward_once(x)
# #         output2 = self.forward_once(y)
# #         return output1, output2



class CNNLayerNorm(nn.Module):
    '''Layer normalization'''
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x : bs, 2, 128, length
        '''|x| = (batch, channel, feature, length)'''
        x = x.transpose(2, 3).contiguous() # (batch, channel, length, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, length) 


class ResidualCNN(nn.Module):
    '''
        Residual CNN
    '''
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, length)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, length)


class BidirectionalGRU(nn.Module):
    '''GRU Block'''
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(2, 32, 3, stride=stride, padding=3//2)  # cnn으로 heirachal특성을 추출

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class, bias=True)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
        x = x.transpose(1, 2) # (batch, length, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = x.mean(dim=-2)
        x = self.classifier(x)
        return x.squeeze(-1)





# triplemarginloss
class SpeechRecognitionModelShamCosine(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super().__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(2, 32, 3, stride=stride, padding=3//2)  # cnn으로 heirachal특성을 추출

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim/2, rnn_dim/4),  
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # change 1000
        self.linear = nn.Sequential(
            nn.Linear(256000/8, 128),
            nn.GELU()
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
        x = x.transpose(1, 2) # (batch, length, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        x = torch.flatten(x,1) # 64, 256000 : rnn_dim / 4
        print(x.shape)
        return x

    def forward(self, x, y):
        output1 = self.forward_once(x)
        print(output1.shape)
        output2 = self.forward_once(y)

        concat = torch.cat((output1,output2), dim = -1)
        output = self.linear(concat)

        return output


class SpeechRecognitionModelFlatten(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(2, 32, 3, stride=stride, padding=3//2)  # cnn으로 heirachal특성을 추출

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class, bias=True)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
        x = x.transpose(1, 2) # (batch, length, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = x.mean(dim=-2)
        x = self.classifier(x)
        return x.squeeze(-1)


class SpeechRecognitionModelSham(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn으로 heirachal특성을 추출

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, rnn_dim)
        )

        self.sequential = nn.Sequential(
            nn.Linear(something*2, something*1/2), # start was 256000
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(something*1/2, 1)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # (batch, feature, length)
        x = x.transpose(1, 2) # (batch, length, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


    def forward(self, x, y):
        output1 = self.forward_once(x).flatten()
        print(output1.shape)
        output2 = self.forward_once(y).flatten()

        concat = torch.cat((output1,output2), dim = -1)
        output = self.sequential(concat)

        return output
