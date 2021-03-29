import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import time

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, sequence_length=10, nonlinearity='relu', dropout=0.0, net_type="LSTM"):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.net_type = net_type
        self.rnn = None
        if net_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity=nonlinearity, dropout=dropout).double()
        elif net_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout).double()
        elif net_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout).double()
        else:
            raise Exception("The net_type has to be either a 'RNN', 'GRU', or 'LSTM")
        self.fc = nn.Linear(hidden_dim*sequence_length, output_size).double()
        self.device = device
    
    def forward(self, x):
        
        batch_size = x.size(0)

        out = None
        #print("layer 2", self.device)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).double().to(self.device)
        if self.net_type == "LSTM":
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).double().to(self.device)
            out, _ = self.rnn(x.double(), (h0.double(), c0.double()))
        else:
            out, _ = self.rnn(x.double(), h0.double())

        out = out.reshape(out.shape[0], -1).double()
        out = self.fc(out.double()).double()

        return out

class RNNet:

    def __init__(self, input_length, output_length, hidden_length, internal_lengths=1, dev="cpu", sequence_length=10, nonlinearity='relu', dropout=0.0, net_type="LSTM", name="model"):

        self.input_length = input_length
        self.output_length = output_length
        self.hidden_length = hidden_length
        self.internal_lengths = internal_lengths
        self.device = dev
        self.cuda_avail = dev=="cuda"
        self.net = RNN(input_length, output_length, hidden_length, internal_lengths, dev, sequence_length=sequence_length, nonlinearity=nonlinearity, dropout=dropout, net_type=net_type).double().to(dev)

        self.loss_function = nn.MSELoss(reduction='mean')
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.name = name + "-" + str(time.time())

        self.training = False
    
    def change_device(self):
        
        if (self.device == "cpu" or self.training) and self.cuda_avail:
            self.device = "cuda"
        elif self.device == "cuda" or not self.training:
            self.device = "cpu"
        
        if self.training:
            self.net.train()
        else:
            self.net.eval()

        #print("layer 1", self.device)
        self.net.device = self.device
        self.net = self.net.to(self.device)
        self.net.device = self.device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #self.net.flatten_parameters()

    def forward(self, x):

        if self.device != "cpu" or self.training:
            self.training = False
            self.change_device()

        if not torch.is_tensor(x):
            x = torch.tensor(x)
        
        result = self.net(x)
        return result
    
    def train_net(self, input, output, batch_size=8, epochs=4):
        
        if (self.device != "cuda" and self.cuda_avail) or not self.training:
            self.training = True
            self.change_device()
            
        if not torch.is_tensor(input):
            input = torch.tensor(input).to(self.device).double()
        
        if not torch.is_tensor(output):
            output = torch.tensor(output).to(self.device).double()


        for epoch in range(epochs):
            #losses = []
            for i in range(0, len(input), batch_size):
                batch_X = input[i:min(i+batch_size, len(input))]
                batch_Y = output[i:min(i+batch_size, len(input))]

                self.optimizer.zero_grad()

                result = self.net(batch_X)
                loss = self.loss_function(result, batch_Y)
                #losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            #print(epoch, sum(losses)/len(losses))
    

    def prep_training_data(self, data):
        
        if (self.device != "cuda" and self.cuda_avail) or not self.training:
            self.training = True
            self.change_device()
        
        return torch.tensor(data).to(self.device).double()

"""
    '''Simple forward pass of the neural network whose value return varies based on whether training is being conducted and an output is provided.'''
    def forward(self, x, y=None, train=False):

        if self.cuda_avail and y is None:
                self.device = self.net.device = "cpu"
                self.net.to(self.device)

        x = x.to(self.device) if torch.is_tensor(x) else torch.tensor(x).to(self.device)

        if y is not None:
            y = y.to(self.device) if torch.is_tensor(y) else torch.tensor(y).to(self.device)
            y = y.double()
            if train:
                self.net.train()
                self.net.zero_grad()
            else:
                self.net.eval()
        else:
            self.net.eval()

        self.net.rnn.flatten_parameters()
        output = self.net(x)
        
        if y is not None:
            output = output.to(self.device)
            loss = self.loss_function(output, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss
        
        if self.cuda_avail:
            self.device = self.net.device = "cuda"
            self.net.to(self.device)

        return output

    '''Train the neural network and perform intermediaate tests if desired. The intermediate testing data is assumed to be different (and new) from the training data.'''
    def train_net(self, input, output, batch_size=8, epochs=4):

        with open("out", 'a') as file:
            for epoch in range(epochs):
                losses = []
                for i in range(0, len(input), batch_size):
                    batch_X = input[i:min(i+batch_size, len(input))]
                    batch_Y = output[i:min(i+batch_size, len(input))]
                    loss = self.forward(batch_X, batch_Y, train=True)
                    losses.append(loss.item())
                #file.write(str(sum(losses)/len(losses)) + ","
"""