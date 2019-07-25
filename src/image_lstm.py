# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 01:27:12 2019

@author: abhii

This script contains an (A)CNN-(B)LSTM-(C)safemapper network.
The CNN encodes the visual information, the LSTM evolves its hidden state which is used by the Classifier to come up with a safe-map of the arena.

"""


import torch.nn.functional as F
#import rospy
#check-gpu

import torch 
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def tensor(obj):
    return torch.tensor(obj, device=device)

class Classifier(nn.Module):

    def __init__(self):
        
        super(Classifier, self).__init__()
        self.counter = 0
        self.hidden_rep_dim = 512
        self.input_feature_dim = 128
        
        #pretrained CNN 
        self.CNN = (models.mobilenet_v2(pretrained=True)).to(device)
        for child in self.CNN.children():
            for param in child.parameters():
                param.requires_grad =False        
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
        
        self.images = []
        #(x,y) - for all points - to be used as embedding
        self.xy_current = []
        
        #(x,y) - at probe points -to be used as classification 
        self.common_1 = nn.Linear(640, 256)
        self.common_1_bn = nn.BatchNorm1d(256)
        self.common_2 = nn.Linear(256, 128)
        self.common_2_bn = nn.BatchNorm1d(128)
        self.common_3 = nn.Linear(128, 64)
        self.common_3_bn = nn.BatchNorm1d(64)
        self.common_4 = nn.Linear(64, 32)
        self.common_4_bn = nn.BatchNorm1d(32)
        self.final = nn.Linear(32, 1)
        self.encoding_embed1 = nn.Linear(int(self.CNN.classifier[-1].out_features), 256)
        self.encoding_xy = nn.Linear(2, 64)
        self.encoding_hybrid = nn.Linear(320, self.input_feature_dim)        
        
        self.optimizer = torch.optim.Adam(self.parameters(),weight_decay = 0.00001,  lr=0.000001)
        self.to(device)
        self.prev_room_data = []
        self.counter_label_list = []
        print(self.parameters())

    def forward(self, x):
        # Network C
        image_embeddings = F.relu(self.encoding_embed1(self.CNN(torch.cat(self.images).to(device))))
        xy_embeddings = F.relu(self.encoding_xy(torch.cat(self.xy_current).to(device)))
        lstm_input = self.encoding_hybrid(torch.cat((image_embeddings,xy_embeddings),1))
        lstm_input = (lstm_input).view(len(lstm_input), 1, -1)
        print("point lstm input len: ", len(lstm_input)) 
        #Network B
        _ , self.embedding = self.lstm(lstm_input,self.hidden)
        
        #Network A
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
        print('probe_complete')
        return torch.cat((y1, y2),1)
        
### untested function`
    def forward_batch(self, Data, x, N):
        print("batch data sizes ", Data[1])
        x1 = F.relu(self.fc_xy1(x))
        hidden_1 = torch.cat([self.hidden[0]]*N,1)
        hidden_2 = torch.cat([self.hidden[1]]*N,1)
        _, embedding = self.lstm(Data,(hidden_1,hidden_2))
        embedding = torch.cat(embedding,2).squeeze(0)
        x2 = F.relu(self.fc_hc1(embedding))

        x = torch.cat((x2, x1),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(y)))
        y = F.relu(self.common_3_bn(self.common_3(y)))
        y = F.relu(self.common_4_bn(self.common_4(y)))
        y1 = (self.final(y))
        y2 = (-self.final(y))
        return torch.cat((y1, y2),1)

#Propogate through classifier network
    def forward_2(self, x):
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
        
#changed
    def store_image(self,xy,image):
        
        xy = [[float(xy[0]), float(xy[1])]]
        xy = tensor(xy)
        self.counter += 1
        image = tensor(image).float()
        image = image.unsqueeze(0) 
        self.images.append(image)
        self.xy_current.append(xy)
##test scripts
    def train_final_network(self,x,label):
        
        y = [[float(x[0]), float(x[1])]]
        
        y = tensor(y)
        label = np.array([label], dtype = long)
        label = tensor(label)        
        loss_function = nn.BCELoss()
        
        
        x1 = F.relu(self.fc_xy1(y))
        embedding = torch.cat(self.embedding,2).squeeze(0)
        x2 = F.relu(self.fc_hc1(embedding.to(device)))
        x = torch.cat((x2,x1),1)
        y = F.relu(self.common_1_bn(self.common_1(x)))
        y = F.relu(self.common_2_bn(self.common_2(x)))
        y = F.relu(self.common_3_bn(self.common_3(x)))
        y = F.relu(self.common_4_bn(self.common_4(x)))
 
        y1 = (self.final(y))
        y1 = F.sigmoid(y1)
        loss = loss_function(y1, label)
        #print(loss.data)#tortch.tensor(1,dtype = torch.long))
        loss.backward(retain_graph = True)
        self.optimizer.step()

    def latent_state(self):

        return(np.array(torch.cat(self.hidden,2).data.squeeze(0).squeeze(0).cpu()))
    
    def get_loss(self,x,label):
        
        with torch.no_grad():
            self.eval()
            self.CNN.eval()
            y = [[float(x[0]), float(x[1])]]
            y = tensor(y)
            label = np.array([label], dtype = long)
            label = tensor(label)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward(y)
            print("point xy", x)
            print("point (get loss) pred",pred)
            loss = loss_function(pred, label)#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_loss(self,x,label):
          
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2_batch(y)
            label = np.array(label, dtype = long)
            label = tensor(label).t()
            loss = loss_function(pred, label)#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))

    def get_batch_confidence(self,x):
        
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=tensor(y)
            pred = self.forward_2_batch(y)

            return(np.array(torch.max(F.softmax(pred),1)[0].cpu()))

    def get_batch_accuracy(self,x,labels):
        
        with torch.no_grad():
            self.eval()
            y = np.array(x,dtype=np.float32)
            y=tensor(y)
            pred = self.forward_2_batch(y)            
            temp = F.softmax(pred)[:,1]>0.5
            correct = (np.array(temp.cpu()) == np.array(labels)) 
            return(float(np.sum(correct))/len(correct))
            
    def get_loss_evolved(self,x,label):
        
        with torch.no_grad():
            self.eval()
            y = [[float(x[0]), float(x[1])]]
            y=tensor(y)
            loss_function = nn.CrossEntropyLoss()
            pred = self.forward_2(y)
            label = np.array([label], dtype = long)
            label = tensor(label)       
            loss = loss_function(pred, label)#torch.tensor(1,dtype = torch.long))
            return(float(np.array(loss.data.cpu())))
#changed
    def clear_images(self):
        if self.counter==0:
            return
        self.counter = 0        
        self.prev_room_data.append([torch.cat(self.images),self.xy_current,self.counter_label_list])       
        self.images = []
        self.counter_label_list = []       
        self.xy_current = [] 
    def train_classifier(self, x, label):
        
        self.train()
        self.counter_label_list.append([self.counter,label,x])
        
        y = [[float(x[0]), float(x[1])]]
        y=torch.tensor(y)

        label = np.array([label], dtype = long)
        label = torch.tensor(label)        

        loss_function = nn.CrossEntropyLoss()
        pred = self.forward(y.to(device))
        loss = loss_function(pred, label.to(device))
    
        loss.backward(retain_graph = True)
        self.optimizer.step()
    def add_data_point(self, x, label):
        self.counter_label_list.append([self.counter,label,x])

    
    def train_classifier_current(self):

#        self.counter_label_list = [counter , labels , (x-y) locations]
        N = len(self.counter_label_list)
        if N <= 1:
            return 
        self.train()
        self.CNN.eval()
        image_embeddings = F.relu(self.encoding_embed1(self.CNN(torch.cat(self.images).to(device))))
        xy_embeddings = F.relu(self.encoding_xy(torch.cat(self.xy_current).to(device)))
        embeddings = self.encoding_hybrid(torch.cat((image_embeddings,xy_embeddings),1))

        print('current last CNN output:', self.CNN(self.images[-1])[0:10])
        max_length = self.counter_label_list[-1][0]
        Data = torch.zeros(N,max_length,128)
        Data_lengths = []
        xy_list = []
        labels = []
        for j,info in enumerate(self.counter_label_list[::-1]):
            Data[j][0:info[0]] = embeddings[0:info[0]]
            Data_lengths.append(info[0])
            labels.append(info[1])
            xy_list.append(info[2])
        Data = Data.to(device)
        xy_list = tensor(xy_list)
        labels = tensor(labels).long()
        Data_lengths = tensor(Data_lengths)
        loss_function = nn.CrossEntropyLoss()            
        Data = torch.nn.utils.rnn.pack_padded_sequence(Data, Data_lengths, batch_first=True)
        pred = self.forward_batch(Data, xy_list, N)    
        labels = labels.t()
        loss = loss_function(pred, labels)#torch.tensor(1,dtype = torch.long))
        print("current training loss:",loss.data)        
        print("current training predictions",pred[:10])
        #print("xy_list", xy_list)
        #print("labels", labels)
        with torch.no_grad():
            pred1 = self.forward(Data,xy_list,N)
        print("current training reprediction",pred1[:10])
        loss.backward()
        self.optimizer.step()

#   untested function
    def train_classifier_prev(self):
        print("starting training on prev")
        self.train()
        self.CNN.train()
        
        for epoch in range(10):

            for i in range(min(10,len(self.prev_room_data))):  
 
                print(i)
                print(len(self.prev_room_data)-i-1) 
                room_images,xy_room,counter_label_list = self.prev_room_data[(len(self.prev_room_data)-i-1)]
                #print("room images",len(room_images))
    	        #print("xy_room",len(xy_room))
	        #print("counter_label_list",counter_label_list)
                image_embeddings = F.relu(self.encoding_embed1(self.CNN(room_images.to(device))))
                xy_embeddings = F.relu(self.encoding_xy(torch.cat(xy_room).to(device)))
                embeddings = self.encoding_hybrid(torch.cat((image_embeddings,xy_embeddings),1))
                N = len(counter_label_list)
 
                max_length = counter_label_list[-1][0]
                #Create empty data matrix
                Data = torch.zeros(N,max_length,128)
                Data_lengths = []
                xy_list = []
                labels = []
                for j,info in enumerate(counter_label_list[::-1]):
                    Data[j][0:info[0]] = embeddings[0:info[0]]
                    Data_lengths.append(info[0])
                    labels.append(info[1])
                    xy_list.append(info[2])

                Data = Data.to(device)
                xy_list = tensor(xy_list)
                labels = tensor(labels).long()
                Data_lengths = tensor(Data_lengths)
                loss_function = nn.CrossEntropyLoss()            
                Data = torch.nn.utils.rnn.pack_padded_sequence(Data, Data_lengths, batch_first=True)
                pred = self.forward_batch(Data, xy_list, N)    
                labels = tensor(labels).t()
                loss = loss_function(pred, labels)#torch.tensor(1,dtype = torch.long))
	        print("prev_training _loss",loss.data)        
                print("prev training pred",pred[:10])
                loss.backward()
                self.optimizer.step()    

