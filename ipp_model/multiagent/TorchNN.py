import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import random

class RNN(nn.Module):
    def __init__(self, input_length, output_length, hidden_length, internal_lengths=[]):
        super(RNN, self).__init__()

        self.net_layers = nn.ModuleList()
        self.output_layer = None
        self.hidden_layer = None

        self.hidden_length = hidden_length

        if len(internal_lengths) == 0:
            self.output_layer = nn.Linear(input_length+hidden_length, output_length)
            self.hidden_layer = nn.Linear(input_length+hidden_length, hidden_length)
        else:
            self.net_layers.append(nn.Linear(input_length+hidden_length, internal_lengths[0]))
            for x in range(1, len(internal_lengths)):
                self.net_layers.append(nn.Linear(internal_lengths[x-1], internal_lengths[x]))
            self.output_layer = nn.Linear(internal_lengths[-1], output_length)
            self.hidden_layer = nn.Linear(internal_lengths[-1], hidden_length)

    def forward(self, input, hidden):
        new_in = []
        for i in input:
            new_in.append(i)
        for i in hidden.detach().numpy():
            new_in.append(i)
        x = torch.tensor(new_in)
        
        for layer in self.net_layers:
            x = F.relu(layer(x),inplace=True)
        hidden = self.hidden_layer(x)
        output = F.relu(self.output_layer(x))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_length).double()

class RNNet:

    def __init__(self, input_length, output_length, hidden_length, internal_lengths=[], dev="cpu", name="model"):

        self.input_length = input_length
        self.output_length = output_length
        self.hidden_length = hidden_length
        self.internal_lengths = internal_lengths
        self.device = torch.device(dev)
        self.net = RNN(input_length, output_length, hidden_length, internal_lengths).double()

        self.loss_function = nn.MSELoss(reduction='mean').double()
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.name = name + "-" + str(time.time())
    
    '''Simple forward pass of the neural network whose value return varies based on whether training is being conducted and an output is provided.'''
    def forward(self, x, y=None, train=False):

        if y is not None:
            y.to(self.device)
            if train:
                self.net.train()
                self.net.zero_grad()
            else:
                self.net.eval()
        else:
            self.net.eval()

        
        hidden = self.net.initHidden()

        for val in x:
            output, hidden = self.net(val, hidden)
        output.to(self.device)
        
        if y is not None:
            #print("------------")
            #print("Out = " + str(output))
            #print("------------")
            #print("Y = " + str(y))
            #print("------------")
            loss = self.loss_function(output.float(), y.float())
            #print("Loss = " + str(loss))
            if train:
                #self.optimizer.zero_grad()
                loss.backward()
                for p in self.net.parameters():
                    p.data.add_(p.grad.data, alpha=-self.learning_rate)
                #self.optimizer.step()

        return output

    '''Train the neural network and perform intermediaate tests if desired. The intermediate testing data is assumed to be different (and new) from the training data.'''
    def train_net(self, input, output, batch_size=8, epochs=4):

        x = input #if torch.is_tensor(input) else torch.tensor(input, dtype=torch.double)
        y = output #if torch.is_tensor(output) else torch.tensor(output, dtype=torch.double)

        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                batch_X = x[i:i+batch_size]
                batch_Y = y[i:i+batch_size]
                print("\n", batch_X, "|", batch_Y, "\n")
                self.forward(torch.tensor(batch_X[0]), torch.tensor(batch_Y[0]), train=True)