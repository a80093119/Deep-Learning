#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from PIL import Image
from collections import Counter
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import os
import sys
from torchvision import transforms
from torch.optim import Adam
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import pycountry
from more_itertools import locate


# In[7]:


sys.modules['torch']


# In[8]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Deep Learning HW2

# In[9]:


data = pd.read_csv('covid_19.csv', header=None)
data = data.drop([1,2], axis=1).drop([1,2])
dataT = data.T.iloc[1:,1:]
dataT.columns = data.iloc[1::,0]
dataT = pd.concat([data.T.iloc[1:,0], dataT.astype(int)], axis=1)
dataT.head()


# In[10]:


crease = pd.DataFrame(np.array(data.iloc[1:,2:].astype(int))-np.array(data.iloc[1:,1:-1].astype(int))).T
crease.columns = data.iloc[1:,0]
crease.tail()


# ## 1.1

# In[11]:


corheatmap = crease.corr()
corheatmap
sns.heatmap(corheatmap, cmap='rainbow')


# ## 1.2

# In[12]:


country = []
for i in np.arange(1, len(corheatmap)):
    for j in range(i):
        if corheatmap.iloc[i][j]>0.7:
            country.append(corheatmap.columns[i])
            country.append(corheatmap.columns[j])
c = list(Counter(country))
subseq = dataT[c] #start from 1/22


# In[13]:


label = copy.copy(crease)
label[crease>0] = 1
label[crease<=0] = 0
label['date'] = np.array(data.iloc[0,2:]) #start from 1/23


# In[43]:


L = 40
start_index = 38 #start from 3/1
for i in range(crease.shape[0]-start_index-L+1):
    if i==0:
        total_subseq = subseq.iloc[start_index+i:start_index+L+i,:].T.values
        total_labels = label[c].iloc[start_index+L,:].values
        
    else:
        total_subseq = np.vstack((total_subseq, subseq.iloc[start_index+i:start_index+L+i,:].T.values))
        total_labels = np.append(total_labels, label[c].iloc[start_index+L,:].values)
end = crease.shape[0]-start_index-L+1
pred_subseq = torch.FloatTensor(dataT.drop(0, axis=1).iloc[start_index+end:start_index+L+end,:].T.values)[:,:,np.newaxis]


# In[15]:


train_data, test_data, train_labels, test_labels = train_test_split(total_subseq, total_labels, test_size=0.3, random_state=42)
train_data = torch.FloatTensor(train_data[:,:,np.newaxis])
test_data = torch.FloatTensor(test_data[:,:,np.newaxis])
train_labels = torch.LongTensor(train_labels)
test_labels = torch.LongTensor(test_labels)
test_labels.shape


# # 1.3

# In[79]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(      #一層RNN
            input_size=1,
            hidden_size=32,     # RNN hidden unit
            num_layers=1,       # RNN layers
            batch_first=True,   #(batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 2)

    def forward(self, x):  
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        h_state = None # 要使用初始 hidden state, 可以设成 None
        
        r_out, h_state = self.rnn(x, h_state)   # h_state 也要作为 RNN 的一个输入
        output = self.out(r_out[:,-1,:])
        
        return output


rnn = RNN()
print(rnn)


# In[16]:


def run_rnn(n_epoch, batch_size, train_data, train_labels, test_data, test_labels, model):
    cross_entropy = []
    train_acc_rate = []
    test_acc_rate = []
    
    for epoch in range(n_epoch):
        train_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        index = torch.randperm(train_data.shape[0])
        train_data = train_data[index]
        train_labels = train_labels[index]

        for batch in range(int(train_data.shape[0]/batch_size)):
            data = train_data[batch*batch_size:(batch+1)*batch_size]
            labels = train_labels[batch*batch_size:(batch+1)*batch_size]
            
            train_output = model(data)   

            loss = loss_func(train_output, labels)     # cross entropy loss
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss += loss.item()*data.shape[0]
            _, train_pred = torch.max(train_output.data, 1)
            train_acc += torch.sum(train_pred == labels.data)
        cross_entropy.append(train_loss/train_data.shape[0])
        train_acc_rate.append(train_acc/train_data.shape[0])
        
        ##test
        model.eval()
        test_output = model(test_data)
        # convert output probabilities to predicted class
        _, test_pred = torch.max(test_output.data, 1)
        # compare predictions to true label
        test_acc += torch.sum(test_pred == test_labels.data)
        test_acc_rate.append(test_acc/test_data.shape[0])
        print('Epoch = %d, train_loss = %f, train_acc = %f, test_acc= %f' %               (epoch, train_loss/train_data.shape[0], train_acc/train_data.shape[0], test_acc/test_data.shape[0]))
        
    
    return cross_entropy, train_acc_rate, test_acc_rate


# In[11]:


np.random.seed(5)
n_epoch = 500
batch_size = 64
learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all rnn parameters
loss_func = nn.CrossEntropyLoss()


# In[82]:


cross_entropy, train_acc_rate, test_acc_rate = run_rnn(n_epoch, batch_size, train_data, train_labels, test_data, test_labels, rnn)


# In[83]:


fig=plt.figure(figsize=(12,4))
##
plt.subplot(121) 
plt.plot(range(n_epoch), train_acc_rate)
plt.xlabel("Epoch")
plt.ylabel("train_acc_rate")
plt.title('train_acc_rate')
##
plt.subplot(122) 
plt.plot(range(n_epoch), test_acc_rate)
plt.xlabel("Epoch")
plt.ylabel("test_acc_rate")
plt.title('test_acc_rate')

plt.show()


# # 1.4

# In[17]:


class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()

        self.lstm = nn.LSTM(     
            input_size=1,      
            hidden_size=64,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # 改以batch在前面 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 2)    # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)   # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        return out

lstm = lstm()
print(lstm)


# In[18]:


np.random.seed(5)
n_epoch = 500
batch_size = 64
learning_rate = 0.001
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)   # optimize all rnn parameters
loss_func = nn.CrossEntropyLoss()


# In[19]:


cross_entropy, train_acc_rate, test_acc_rate = run_rnn(n_epoch, batch_size, train_data, train_labels, test_data, test_labels, lstm)


# In[20]:


fig=plt.figure(figsize=(12,4))
##
plt.subplot(121) 
plt.plot(range(n_epoch), train_acc_rate)
plt.xlabel("Epoch")
plt.ylabel("train_acc_rate")
plt.title('train_acc_rate')
##
plt.subplot(122) 
plt.plot(range(n_epoch), test_acc_rate)
plt.xlabel("Epoch")
plt.ylabel("test_acc_rate")
plt.title('test_acc_rate')

plt.show()


# # 1.5

# In[53]:


import torch.nn.functional as F
pred_output = lstm(pred_subseq) 
_, pred = torch.max(pred_output.data, 1)


# In[58]:


# 將國家轉換為2個音文字母的代碼
country_set = crease.columns
input_countries = country_set.tolist()
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2
codes = [countries.get(country, 'Unknown code') for country in input_countries] 
#找到對應的'Unknown code'正確代碼
indices = list(locate(codes, lambda x: x == 'Unknown code'))
subcode = []
unknown = []
for i in range(len(indices)):
    try:
        subcode.append(pycountry.countries.lookup(country_set[indices][i]).alpha_2)
    except:
        unknown.append(country_set[indices][i])
subcode = ['BO', 'BN', 'BU', 'CG', 'CD', 'CI', 'Diamond Princess', 'VA', 'IR', 'KR', 'kosovo', 'LA', 'MS Zaandam', 'MD', 'RU', 'SY', 'TW', 'TZ', 'us', 'VE', 'VN', 'West Bank and Gaza']
for i in range(len(indices)):
    codes[indices[i]] = subcode[i]
country_code = np.array([x.lower() for x in codes])


# In[55]:


pred_1 = country_code[list(locate(pred_label, lambda x: x == 1))].tolist()
pred_0 = country_code[list(locate(pred_label, lambda x: x == 0))].tolist()


# In[56]:


import pygal_maps_world.maps
worldmap_chart = pygal_maps_world.maps.World()
worldmap_chart.title = 'Covid_19'
worldmap_chart.add('Ascending', pred_1)
worldmap_chart.add('Descending', pred_0)


# # VAE

# # 2.1

# In[40]:


np.random.seed(5)
##read picture
data_images = []
for i in range(21551):
    filepath = 'data/'
    filename = str((i+1)) +'.png'
    image = np.array(Image.open(filepath+filename).convert('RGB'))/255
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    image = np.moveaxis(image, 2, 0)
    data_images.append(image)
data_images = torch.from_numpy(np.array(data_images))
data_images


# # 2.2~2.4

# In[58]:


# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=3*28*28, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    # 编码过程
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


# In[59]:


image_size = 3*28*28
h_dim = 400
z_dim = 50
model = VAE(image_size=image_size, h_dim=h_dim, z_dim=z_dim).to(device)
num_epochs = 200
batch_size = 64
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss(reduction='sum')
train_data = data_images.float()


# In[60]:


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[61]:


elbo = []
for epoch in range(num_epochs):
    index = torch.randperm(train_data.shape[0])
    train_data = train_data[index]
    for batch in range(int(train_data.shape[0]/batch_size)+1):
        if batch == range(int(train_data.shape[0]/batch_size)+1)[-1]:
            x = train_data[batch*batch_size:]
        else:
            x = train_data[batch*batch_size:(batch+1)*batch_size]

        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)
         
        reconst_loss = loss_fn(x_reconst, x)
        kl_div = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        loss = reconst_loss + 0*kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch+1) % 100 == 0:
            print ("Epoch = %d, loss = %f " %(epoch+1, loss)
    elbo.append(loss.item())
        
    if (epoch+1) % 20 == 0 :
        with torch.no_grad():
            # 隨機生成
            z = torch.randn(batch_size, z_dim)
            out = model.decode(z).view(-1, 3, 28, 28)
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

            # 重構
            out, _, _ = model(x)
            x_concat = torch.cat([x.view(-1, 3, 28, 28), out.view(-1, 3, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))


# In[62]:


plt.plot(elbo)


# # 2.5

# In[63]:


two_pic = data_images[0:2].float()
two_pic = two_pic.to(device).view(-1, image_size)
out, _, _ = model(two_pic)
out = out.view(-1, 3, 28, 28)
pic = two_pic.view(-1, 3, 28, 28)[0]
for i in np.arange(0.1, 1, 0.1):
    pic = torch.cat([pic, (1-i)*out[0]+i*out[1]], dim=2)
pic = torch.cat([pic, two_pic.view(-1, 3, 28, 28)[1]], dim=2)
save_image(pic, os.path.join(sample_dir, 'two picture.png'))


# In[ ]:




