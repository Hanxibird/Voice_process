import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net_v6_80bark(nn.Module):
    def __init__(self):
        super(Net_v6_80bark, self).__init__()
        self.depth_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)

        self.depth_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, groups=16, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1)

        self.depth_conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, groups=16, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        self.conv33 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        hz = 256
        self.lstm = nn.GRU(input_size=74, hidden_size=hz, num_layers=1, batch_first=True)
        self.lstm_ = nn.GRU(input_size=hz, hidden_size=hz, num_layers=1, batch_first=True)
        self.lstm__ = nn.GRU(input_size=hz, hidden_size=hz, num_layers=1, batch_first=True)

        #self.linear = nn.Linear(hz * 2, 96)
        self.linear = nn.Linear(hz, 80)

    def forward(self, x_bark):

        #x_bark = torch.matmul(torch.pow(x_bark, 2), self.bin2bark_matrix)
        cnv1__ = self.depth_conv1(x_bark)
        cnv1_ = self.conv1(cnv1__)
        cnv1 = F.relu(cnv1_)
        cnv11_ = self.conv11(cnv1__)
        cnv111 = F.relu6(cnv11_ + 3.0) / 6.0
        cnv1 = torch.mul(cnv1, cnv111)

        cnv2__ = self.depth_conv2(cnv1)
        cnv2_ = self.conv2(cnv2__)
        cnv2 = F.relu(cnv2_)
        cnv22_ = self.conv22(cnv2__)
        cnv222 = F.relu6(cnv22_ + 3.0) / 6.0
        cnv2 = torch.mul(cnv2, cnv222)

        cnv3__ = self.depth_conv3(cnv2)
        cnv3_ = self.conv3(cnv3__)
        cnv3 = F.relu(cnv3_)
        cnv33_ = self.conv33(cnv3__)
        cnv333 = F.relu6(cnv33_ + 3.0) / 6.0
        cnv3 = torch.mul(cnv3, cnv333)

        rnn_input = cnv3.view(cnv3.size(0), cnv3.size(2), -1)

        rnn_output, _ = self.lstm(rnn_input)
        rnn_output, _ = self.lstm_(rnn_output)
        rnn_output, _ = self.lstm__(rnn_output)

        rnn_output = rnn_output.view(rnn_output.size(0), -1, rnn_output.size(1), rnn_output.size(2))

        output = F.relu(self.linear(rnn_output))
        return output


