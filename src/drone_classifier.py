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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pre-trained CNN model - MobileNetV2 on ImageNet
CNN = models.mobilenet_v2(pretrained=True)
CNN.to(device)

class Classifier(nn.Module):

    def __init__(self ,trained_CNN=CNN):
        
        super(Classifier, self).__init__()
        self.counter = 0
        self.hidden_rep_dim = 512
        self.input_feature_dim = 128
        #pretrained CNN
        self.CNN = trained_CNN
        #LSTM random initialization
        self.lstm = nn.LSTM(self.input_feature_dim, self.hidden_rep_dim)        
        self.word_lstm_init_h = nn.Parameter(torch.randn(1, 1, self.hidden_rep_dim).type(torch.FloatTensor), requires_grad=True)
        self.word_lstm_init_c = nn.Parameter(torch.randn(1, 1, self.hidden_rep_dim).type(torch.FloatTensor), requires_grad=True)
        self.hidden = (self.word_lstm_init_h,self.word_lstm_init_c)        
        self.embedding = None
        #Classifier Network
        self.fc_embed1 = nn.Linear(1024, 250)  
        self.fc_xy1 = nn.Linear(2, 50)
#        self.image_embeddings = []
        self.hybrid_image_embeddings = []
        self.xy = []
        self.common_1 = nn.Linear(300, 84)
        self.common_2 = nn.Linear(84,16)
        self.final = nn.Linear(16,1)
        self.encoding_embed1 = nn.Linear(int(CNN.classifier[-1].out_features), 256)
        self.encoding_xy = nn.Linear(2, 64)
        self.encoding_hybrid = nn.Linear(320, self.input_feature_dim)        
        self.to(device)
#Propogate through LSTM
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
        
#Propogate through classifier network
    def forward_2(self, x):
        #If there is no evolution of state
#        if self.embedding == None:
#            return(self.forward(x))
            
        x1 = F.relu(self.fc_xy1(x))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_embed1(embedding))
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1(x))
        y = F.relu(self.common_2(y))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)

    def forward_2_batch(self, x):
        #If there is no evolution of state
#        if self.embedding == None:
#            return(self.forward(x))
            
        x1 = F.relu(self.fc_xy1(x))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_embed1(embedding))
        
        x2 = torch.repeat_interleave(x2,x1.shape[0]).reshape([x2.shape[1],x1.shape[0]]).t()
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1(x))
        y = F.relu(self.common_2(y))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)
        

    def encode_input(self,xy,image):
        
        xy = [float(xy[0]), float(xy[1])]
        xy = torch.tensor(xy)
        self.counter += 1
#        rospy.loginfo(rospy.get_caller_id())
#        image = imgmsg_to_grayscale_array(image)        
        image =torch.tensor(image, device=device).float()
        image = image.unsqueeze(0).unsqueeze(0)  
        image = image.to(device)
        xy  = xy.to(device)
        image = self.encoding_embed1(self.CNN(image))
        image = F.relu(image)
        xy = self.encoding_xy(xy)
        xy = F.relu(xy)
        z = torch.cat((image,xy),1)
        self.hybrid_image_embeddings.append(F.relu(self.encoding_hybrid(z)))
        
    def latent_state(self):
        return(np.array(torch.cat(self.hidden,2).data.squeeze(0).squeeze(0).cpu()))
    
    def get_loss(self,x,label):
        
        with torch.no_grad():
            y = [float(x[0]), float(x[1])]
            y=torch.tensor(y)
            label = np.array([label], dtype = long)
            label = torch.tensor(label)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward(y.to(device))
            print(pred)
    #        target = torch.tensor(np.ones((1),dtype = long))
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_loss(self,x,label):
          
        #   y = [float(x[0]), float(x[1])]
        with torch.no_grad():
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2_batch(y.to(device))
            label = np.array(label, dtype = long)
            label = torch.tensor(label).t()
    #        target = torch.tensor(np.ones((1),dtype = long))
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_confidence(self,x):
        
        #   y = [float(x[0]), float(x[1])]
        with torch.no_grad():
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            pred = self.forward_2_batch(y.to(device))

            return(np.array(torch.max(F.softmax(pred),1)[0].cpu()))

    def get_batch_accuracy(self,x,labels):
        
        with torch.no_grad():
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            pred = self.forward_2_batch(y.to(device))            
            temp = F.softmax(pred)[:,1]>0.5
            correct = (np.array(temp.cpu()) == np.array(labels)) 
            return(float(np.sum(correct))/len(correct))
            
    def get_loss_evolved(self,x,label):
        
        with torch.no_grad():
            y = [float(x[0]), float(x[1])]
            y=torch.tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2(y)
            label = np.array([label], dtype = long)
            label = torch.tensor(label)       
    #        target = torch.tensor(np.ones((1),dtype = long))
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def clear_image_embeddings(self):
        
        self.hybrid_image_embeddings = []
    
    def train_classifier(self, x, label):
        
        
        y = [float(x[0]), float(x[1])]
        y=torch.tensor(y)
        label = np.array([label], dtype = long)
        label = torch.tensor(label)        
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        pred = self.forward(y.to(device))
#        target = torch.tensor(np.ones((1),dtype = long))
        loss = loss_function(pred, label.to(device))
        print(loss.data)#torch.tensor(1,dtype = torch.long))
        loss.backward(retain_graph = True)
        optimizer.step() 


#D1 = Classifier(CNN)
#for i in range(12):    
##    #image = plt.imread('/home/abhi/pytorch_1/data/'+str(284+i*10)+'.jpg')
 #   image  = i*np.ones((480,752))    
 #   D1.encode_input([1.2,1.3],image)

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
