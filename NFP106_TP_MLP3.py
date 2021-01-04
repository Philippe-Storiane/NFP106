# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:12:11 2020

@author: a179415
"""

import numpy as np
import matplotlib.pyplot as plt

# from tqdm import tqdm

import torch
import torch.nn as nn


input_dim = 12
hidden_dim = 10
output_dim = 1

years = np.loadtxt("Sunspots", usecols=0, dtype=int)
values = np.loadtxt("Sunspots", usecols=1)
values_size = values.shape[0]
values = 2 * values - 1
x = np.zeros( (values_size - input_dim, input_dim ))
for i in range( input_dim ):
    x[:, i] = values[ i:  (values_size - input_dim + i)]
variance_data_sunspots = np.sum(np.power(x - np.mean(x), 2)) / x.shape[0]
x = torch.FloatTensor( x )
y = values[input_dim : values_size]
y = torch.FloatTensor( y )

x_train = x[0:209,:]
y_train = y[0:209]
x_val = x[209:244,:]
y_val = y[ 209:244]
x_test = x[244:268, :]
y_test = y[244:268]

#fig = plt.figure()
#plt.plot( years,values, 'o-k')
#plt.xlabel("year")
#plt.ylabel("Average sunspot activity")
#plt.title("Série sunspot normalisée de 1700 `a 1979")
#plt.show()


class Model(nn.Module):
    
    def __init__(self, iniut_dim, hiddem_dim, output_dim, lr ):
        super(Model, self).__init__()
        self.layer1 = nn.Linear( input_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        parameters = [ p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD( parameters, lr = lr)
    
    def forward(self, input):
        hidden = self.layer1( input)
        hidden = torch.tanh( hidden)
        output = self.layer2( hidden )
#        output = torch.tanh( output)
        return output

    
def train(model, x_data, y_data, n_epochs):
    model.train()
    loss = None
    for epoch in range(n_epochs):
#        tqdm(total=len(x_data),unit_scale=True,postfix={'loss':0.0,'test loss':0.0},
#                      desc="Epoch : %i/%i" % (epoch, n_epochs-1),ncols=100) as pbar:
        model.optimizer.zero_grad()
        x_pred = model.forward( x_data )
        loss = model.criterion(x_pred.squeeze() , y_data)
        loss.backward()
        model.optimizer.step()
        if epoch % 100 == 0:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    return loss

def eval_model(model, x_data, y_data):
    model.eval()
    output = model.forward(x_data)
    loss = model.criterion( output.squeeze(), y_data)
    print("eval loss {}".format(loss.item() ))
    return loss

lr = 0.01
list_epochs = [5, 10,15, 20,25, 100, 200, 300,400,500,800,900,1000,1100, 1200, 2000,3000,4000,5000]
ARV_train=[]
ARV_val=[]
ARV_test=[]
for epochs in list_epochs:    
    model = Model( input_dim, hidden_dim, output_dim, lr)
    print("training life cyle {}".format(epochs))
    loss = train(model, x_train, y_train, epochs)
    ARV_train.append( loss.item() / variance_data_sunspots)
    print('ARV training {}'.format( ARV_train[-1]))
    print('validating')
    loss=eval_model(model, x_val, y_val)
    ARV_val.append( loss.item() / variance_data_sunspots)
    print('ARV val {}'.format( ARV_val[-1]))
    print('testing')
    loss=eval_model(model, x_test, y_test)
    ARV_test.append( loss.item() / variance_data_sunspots)
    print('ARV testing {}'.format( ARV_test[ -1 ]))
    
fig = plt.figure()
plt.plot(list_epochs, ARV_train, label="train")
plt.plot(list_epochs, ARV_val, label="val")
plt.plot(list_epochs, ARV_test, label="test")
plt.xlabel("life cycle")
plt.ylabel('ARV')
plt.legend()
plt.show()


def display(epoch):
    index = list_epochs.index(epoch)
    print("Train {}".format(ARV_train[ index ]))
    print("Val {}".format(ARV_val[ index ]))
    print("Test {}".format(ARV_test[ index ]))