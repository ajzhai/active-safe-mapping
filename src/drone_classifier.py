# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 01:27:12 2019

@author: abhi
"""

import torch 
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
#import rospy
#check-gpu
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#pre-trained CNN model - Alexnet
CNN = models.alexnet()
CNN.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
CNN.to(device)
#pre-trained CNN model - 
#CNN = models.inc()
#CNN.Conv2d_1a_3x3 = models.inception.BasicConv2d(1, 32, kernel_size=3, stride=2)

for child in CNN.children():
    for param in child.parameters():
        param.requires_grad =False

        
class Classifier(nn.Module):

    def __init__(self ,trained_CNN=CNN):
        super(Classifier, self).__init__()
        self.counter = 0
        self.hidden_rep_dim = 1002
        self.input_fetaure_dim = 1002
        #pretrained CNN
        self.CNN = trained_CNN
        #LSTM random initialization
        self.lstm = nn.LSTM(self.hidden_rep_dim,self.input_fetaure_dim)        
        self.embedding = None
        #Classifier Network
        self.fc_embed1 = nn.Linear(2004, 120)  
        self.fc_xy1 = nn.Linear(2,10)
#        self.image_embeddings = []
        self.hybrid_image_embeddings = []
        self.xy = []
        self.common_1 = nn.Linear(130, 84)
        self.common_2 = nn.Linear(84,16)
        self.word_lstm_init_h = nn.Parameter(torch.randn(1, 1, self.input_fetaure_dim).type(torch.FloatTensor), requires_grad=True)
        self.word_lstm_init_c = nn.Parameter(torch.randn(1, 1, self.hidden_rep_dim).type(torch.FloatTensor), requires_grad=True)
        self.hidden = (self.word_lstm_init_h,self.word_lstm_init_c)        
        self.final = nn.Linear(16,1)
        

    def forward(self, x):
        lstm_input = torch.cat(self.hybrid_image_embeddings).view(len(self.hybrid_image_embeddings), 1, -1)
        
        _ , self.embedding = self.lstm(lstm_input,self.hidden)
        x1 = F.relu(self.fc_xy1(x))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_embed1(embedding))
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1(x))
        y = F.relu(self.common_2(y))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)

    def save_image_embedding(self,xy,image):
        xy_new = []
        xy_new.append(float(xy[0]))
        xy_new.append(float(xy[1]))
        xy = [xy_new]
        xy = torch.tensor(xy)
        self.counter += 1
#        rospy.loginfo(rospy.get_caller_id())
#        image = imgmsg_to_grayscale_array(image)        
        image =torch.tensor(image, device=device).float()
        image = image.unsqueeze(0).unsqueeze(0)  
        self.hybrid_image_embeddings.append(torch.cat((self.CNN(image),xy.to(device)),1))
        
    def latent_state(self):
        return(np.array(torch.cat(self.hidden,2).data.squeeze(0).squeeze(0)))
    
    def get_loss(self,x,label):
        y = []
        y.append(float(x[0]))
        y.append(float(x[1]))
        y=[y]
        y=torch.tensor(y)
        label = np.array([label], dtype = long)
        label = torch.tensor(label)
        loss_function = nn.CrossEntropyLoss()
        pred = self.forward(y.to(device))
        print(pred)
#        target = torch.tensor(np.ones((1),dtype = long))
        loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
        return(float(np.array(loss.data)))

    def clear_image_embeddings(self):
        self.hybrid_image_embeddings = []
    
    def train_classifier(self, x, label):
        y = []
        y.append(float(x[0]))
        y.append(float(x[1]))
        y=[y]
        y=torch.tensor(y)
        label = np.array([label], dtype = long)
        label = torch.tensor(label)        
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        pred = self.forward(y.to(device))
#        target = torch.tensor(np.ones((1),dtype = long))
        loss = loss_function(pred, label.to(device))
        print(self.latent_state()) 
        print(loss.data)#torch.tensor(1,dtype = torch.long))
        loss.backward()
        optimizer.step() 

#CNN(torch.tensor([[image]]).type(torch.FloatTensor).to(device))
#'''D1.lstm(torch.cat(D1.image_embeddings).view(len(D1.image_embeddings), 1, -1),hidden)
#'''
#D1.train_classifier(torch.tensor(np.ones((1),dtype = long)))
#
#
#for i,j in enumerate(D1.lstm.parameters()):
#    if i==2:
#        B = j.grad
#a
