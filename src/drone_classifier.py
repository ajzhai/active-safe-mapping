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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#pre-trained CNN model - MobileNetV2 on ImageNet
CNN = models.mobilenet_v2(pretrained=True)
CNN.to(device)

class Classifier(nn.Module):

    def __init__(self ,trained_CNN=CNN):
        
        super(Classifier, self).__init__()
        self.counter = 0
        self.hidden_rep_dim = 512
        self.input_feature_dim = 128
####        self.counter_label_list = []

        #pretrained CNN
        self.CNN = trained_CNN
        #LSTM random initialization
        self.lstm = nn.LSTM(self.input_feature_dim, self.hidden_rep_dim)        
        self.word_lstm_init_h = nn.Parameter(torch.randn(1, 1, self.hidden_rep_dim).type(torch.FloatTensor), requires_grad=True)
        self.word_lstm_init_c = nn.Parameter(torch.randn(1, 1, self.hidden_rep_dim).type(torch.FloatTensor), requires_grad=True)
        #learnable initial hidden state
        self.hidden = (self.word_lstm_init_h,self.word_lstm_init_c)        
        #latest evolved hidden state
        self.embedding = None
        #Classifier Network
        self.fc_hc1 = nn.Linear(1024, 512)
        self.fc_xy1 = nn.Linear(2, 128)
        #self.image_embeddings = []
        self.hybrid_image_embeddings = []
        self.xy = []
        self.common_1 = nn.Linear(640, 256)
        self.common_1_bn = nn.BatchNorm1d(256)
        self.common_2 = nn.Linear(256, 128)
        self.common_2_bn = nn.BatchNorm1d(128)
        self.common_3 = nn.Linear(128, 64)
        self.common_3_bn = nn.BatchNorm1d(64)
        self.common_4 = nn.Linear(64, 32)
        self.common_4_bn = nn.BatchNorm1d(32)
        self.final = nn.Linear(32, 1)

        self.encoding_embed1 = nn.Linear(int(CNN.classifier[-1].out_features), 256)
        self.encoding_xy = nn.Linear(2, 64)
        self.encoding_hybrid = nn.Linear(320, self.input_feature_dim)        
        self.to(device)
        self.load_state_dict(torch.load('/home/azav/results/cool_weights.pth'))
####        self.prev_room_data = []

#Propogate through LSTM
    def forward(self, x):
        
        lstm_input = torch.cat(self.hybrid_image_embeddings).view(len(self.hybrid_image_embeddings), 1, -1)
        _ , self.embedding = self.lstm(lstm_input,self.hidden)
        x1 = F.relu(self.fc_xy1(x))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_hc1(embedding))
        x = torch.cat((x2, x1), 1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(y)))
        y = F.relu(self.common_3_bn(self.common_3(y)))
        y = F.relu(self.common_4_bn(self.common_4(y)))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)
### untested function`
    def forward_batch(self, Data, x):
        #If there is no evolution of state
#        if self.embedding == None:
#            return(self.forward(x))
        
        x1 = F.relu(self.fc_xy1(x))
        hidden_1 = torch.cat([self.hidden[0]]*4,1)
        hidden_2 = torch.cat([self.hidden[1]]*4,1)        
#        embedding = torch.cat(self.embedding,2).squeeze(0)
        _, embedding = self.lstm(Data,(hidden_1,hidden_2))
        
        x2 = F.relu(self.fc_hc1(embedding))

        print(x2.shape,x1.shape)
        
#        x2 = torch.repeat_interleave(x2,x1.shape[0]).reshape([x2.shape[1],x1.shape[0]]).t()
        x = torch.cat((x2[0],x1[0]),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(y)))
        y = F.relu(self.common_3_bn(self.common_3(y)))
        y = F.relu(self.common_4_bn(self.common_4(y)))
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
        x2 = F.relu(self.fc_hc1(embedding))
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(y)))
        y = F.relu(self.common_3_bn(self.common_3(y)))
        y = F.relu(self.common_4_bn(self.common_4(y)))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)

    def forward_2_batch(self, x):
            
        x1 = F.relu(self.fc_xy1(x))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_hc1(embedding.to(device)))
        x2 = torch.repeat_interleave(x2,x1.shape[0]).reshape([x2.shape[1],x1.shape[0]]).t()
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(y)))
        y = F.relu(self.common_3_bn(self.common_3(y)))
        y = F.relu(self.common_4_bn(self.common_4(y)))
        y1 = (self.final(y))
        y2 = (-self.final(y))

        return torch.cat((y1, y2),1)
        

    def encode_input(self,xy,image):
        
        xy = [[float(xy[0]), float(xy[1])]]
        xy = torch.tensor(xy)
        self.counter += 1
        image =torch.tensor(image, device=device).float()
        image = image.unsqueeze(0) 
        image = image.to(device)
        xy  = xy.to(device)
        image = self.encoding_embed1(self.CNN(image))
        image = F.relu(image)
        xy = self.encoding_xy(xy)
        xy = F.relu(xy)
        z = torch.cat((image,xy),1)
        self.hybrid_image_embeddings.append(F.relu(self.encoding_hybrid(z)))

##test scripts
    def train_final_network(self,x,label):

        self.train()
        y = [[float(x[0]), float(x[1])]]
        y=torch.tensor(y)
        label = np.array([label], dtype = long)
        label = torch.tensor(label)        
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        x1 = F.relu(self.fc_xy1(y.to(device)))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_hc1(embedding.to(device)))
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(x)))
        y = F.relu(self.common_3_bn(self.common_3(x)))
        y = F.relu(self.common_4_bn(self.common_4(x)))
        y1 = (self.final(y))
        y1 = F.sigmoid(y1)
        loss = loss_function(y1, label.to(device))
        print(loss.data)#torch.tensor(1,dtype = torch.long))
        loss.backward(retain_graph = True)
        optimizer.step()



       


    def latent_state(self):
        return(np.array(torch.cat(self.hidden,2).data.squeeze(0).squeeze(0).cpu()))
    
    def get_loss(self,x,label):
        
        with torch.no_grad():
            self.eval()
            y = [[float(x[0]), float(x[1])]]
            y=torch.tensor(y)
            label = np.array([label], dtype = long)
            label = torch.tensor(label)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward(y.to(device))
            print(pred)
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_loss(self,x,label):
          
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2_batch(y.to(device))
            label = np.array(label, dtype = long)
            label = torch.tensor(label).t()
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_confidence(self,x):
        
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            pred = self.forward_2_batch(y.to(device))

            return(np.array(torch.max(F.softmax(pred),1)[0].cpu()))

    def get_batch_accuracy(self,x,labels):
        
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=torch.tensor(y)
            pred = self.forward_2_batch(y.to(device))            
            temp = F.softmax(pred)[:,1]>0.5
            correct = (np.array(temp.cpu()) == np.array(labels)) 
            return(float(np.sum(correct))/len(correct))
            
    def get_loss_evolved(self,x,label):
        
        with torch.no_grad():
            self.eval()
            y = [[float(x[0]), float(x[1])]]
            y=torch.tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2(y)
            label = np.array([label], dtype = long)
            label = torch.tensor(label)       
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def clear_image_embeddings(self):
        self.counter = 0
    ####     self.prev_room_data.append([self.hybrid_image_embeddings,self.counter_label_list])       
    ####     print(len(self.prev_room_data))
        self.hybrid_image_embeddings = []
    ####    self.counter_label_list = []       
    
    def train_classifier(self, x, label):
        self.train()
    ###    self.counter_label_list.append([self.counter,label,self.xy])
    ###    print(len(self.counter_label_list))
        y = [[float(x[0]), float(x[1])]]
        y=torch.tensor(y)
        label = np.array([label], dtype = long)
        label = torch.tensor(label)        
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        pred = self.forward(y.to(device))
        loss = loss_function(pred, label.to(device))
        print(loss.data)#torch.tensor(1,dtype = torch.long))
        loss.backward(retain_graph = True)
        optimizer.step()



#   untested function
    def train_classifier_prev(self):
        self.train()
        # pointer to the data
        for i in range(min(5,len(self.hybrid_image_embeddings))):  
            x,y = self.prev_room_data[(len(self.hybrid_image_embeddings)-i)]
            max_length = y[-1][0]
            N = len(y)
            Data = torch.zeros(N,max_length,512)
            Data_lengths = []
            xy_list = []
            labels = []
        # data is copied
            
            for j,info in enumerate(y[::-1]):
                Data[j][0:info[0]] = x[0:info[0]]
                Data_lengths.append(info[0])
                labels.append(info[1])
                xy_list.append(info[2])
            xy_list = torch.tensor(xy_list)
            labels = torch.tensor(labels)
            Data_lengths = torch.tensor(Data_lengths)
            loss_function = nn.CrossEntropyLoss()            
            Data = torch.nn.utils.rnn.pack_padded_sequence(Data, Data_lengths, batch_first=True)
            pred = self.forward_batch(Data.to(device), xy_list)    
            label = torch.tensor(label).t()
            print(label,label.shape)
    #        target = torch.tensor(np.ones((1),dtype = long))
            loss = loss_function(pred, label.to(device))#torch.tensor(1,dtype = torch.long))
            loss.backward(retain_graph = True)
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
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

