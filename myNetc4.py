import torch
import torch.nn as nn
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
import torch.nn.functional as F




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(2)
        proj_query = self.query_conv(x).permute(0, 2, 1)  # 1 X N X C
        proj_key = self.key_conv(x)  # 1 X C x N
        energy = torch.bmm(proj_query, proj_key)  # 1 X N X N
        attention = self.softmax(energy)  # 1 X N X N

        out = torch.bmm(x, attention.permute(0, 2, 1))
        out = self.gamma * out + x

        return out



class Cnn_With_Clinical_Net(nn.Module):
    def __init__(self, model):
        super(Cnn_With_Clinical_Net, self).__init__()
        # self.layer = nn.Sequential(*list(model.children())[:-1])
        # self.feature = list(model.children())[-1].in_features
        # self.cnn = nn.Linear(self.feature, 128)

        # CNN
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-2]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features

        # LSTM with Attention for both directions
        self.attention1 = SelfAttention(self.feature)  # 添加自注意力机制层
        self.attention2 = SpatialAttention(kernel_size=7)
        self.lstm = nn.LSTM(self.feature, hidden_size=self.feature, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.feature*2, 128)  #with lstm
        # self.linear = nn.Linear(self.feature, 128)   #without lstm

        # clinical feature
        self.clinical = nn.Linear(2, 2)

        # concat
        self.mcb = CompactBilinearPooling(128, 2, 128).cuda()
        # self.mcb = CompactBilinearPooling(128, 1, 128)
        # self.concat = nn.Linear(128+1, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x, clinical_features):
        x = self.conv(x) #torch.Size([128, 2048, 7, 7])
        x1 = self.attention2(x)  #spatial
        # x1 = self.attention3(x)     #CBAM
        x = x*x1
        x = F.adaptive_avg_pool2d(x,(1,1))
        # # resnet.alexnet
        x = x.view(x.size(0),-1)

        if self.dense is not None:
            x = self.dense(x)

        # # iv250
        # # Apply LSTM with Attention for both directions
        # print('1',x.shape) #[4, 2048]
        # x = self.attention_lstm(x)
        # print('6',x.shape)  #[4, 4096]
        # x = self.dropout(x)

        x = self.attention1(x)    #selfattention
        x = x.view(x.size(0), -1)

        # # iv250
        # lstm_out, _ = self.lstm(x)
        # x = lstm_out
        # x = self.linear(x)
        # # iv
        lstm_out, _ = self.lstm(x.view(x.size(0), 1, -1))
        x = self.linear(lstm_out.view(x.size(0),-1))
        # #############

        clinical = self.clinical(clinical_features)
        # print(clinical.shape)
        x = self.mcb(x, clinical)
        # x = torch.cat([x, clinical], dim=1)
        # x = self.concat(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-2]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.attention1 = SelfAttention(self.feature)  # 添加自注意力机制层
        self.attention2 = SpatialAttention(kernel_size=7)
        self.lstm = nn.LSTM(self.feature, hidden_size=self.feature, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.feature*2, 2)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.attention2(x)
        x = x * x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = self.attention1(x)
        x = x.view(x.size(0), -1)
        # iv250
        # lstm_out, _ = self.lstm(x)
        # x = lstm_out
        # x = self.linear(x)
        # iv
        lstm_out, _ = self.lstm(x.view(x.size(0), 1, -1))
        x = self.linear(lstm_out.view(x.size(0), -1))

        return x



