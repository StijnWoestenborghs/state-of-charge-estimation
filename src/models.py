import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_inputs):
        super(Net, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 16)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(16, 32)
        self.act2 = nn.ReLU()
        # third hidden layer
        self.hidden3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        # output layer
        self.hidden4 = nn.Linear(16, 1)
        self.act4 = nn.Sigmoid()
 
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X
    
    
    
class DynamicNet(nn.Module):
    def __init__(self, n_inputs, layer_sizes):
        super(DynamicNet, self).__init__()

        # input layer
        layers = []
        layer_sizes.insert(0, n_inputs)  # Insert the input size at the beginning

        # Create the hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.layers = nn.ModuleList(layers)

        # output layer
        self.output_layer = nn.Linear(layer_sizes[-1], 1)
        self.output_act = nn.Sigmoid()

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        
        X = self.output_layer(X)
        X = self.output_act(X)
        return X