#!/usr/bin/env python
# coding: utf-8

# In[131]:


import os
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import imageio

import pandas as pd



# ### Data preprocessing

# In[132]:


# # Read the data and cocatanate the data frame

# # Read two CSV files which includes joint angle and EMG features

# # For joint angle estimation
# path = r'D:\GitHub\EMG_regressive_model\data\2'
# all_files = glob.glob(path + "/*csv")
# print(all_files)

# dfList = []

# for filename in all_files:
#     df = pd.read_csv(filename)
#     df.head()
#     dfList.append(df)
# frame = pd.concat(dfList, axis = 1, ignore_index = False)
# frame.head()


# In[133]:


# For load estimation

path = r'D:\GitHub\EMG_regressive_model\data\2'
all_files = glob.glob(path + "/*csv")
dfList = []

for filename in all_files:
    df = pd.read_csv(filename)
    df.head()
    dfList.append(df)
frame = pd.concat(dfList, axis = 1, ignore_index = False)
frame.head()


# In[134]:


# sns.countplot(x = 'load', data=frame)


# In[135]:


# create input and output data
x = frame.iloc[:, 0:-1]
y = frame.iloc[:, -1]

# split the data into train, validate, and test set
# because there is a class imblance, should use the stratify opton to make each set has identical distribution
X_trainval, X_test, y_trainval, y_test = train_test_split(x, y)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1)


# In[136]:


X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[137]:


# def get_load_distribution(obj):
#     count_dict = {
#         'load_1': 0,
#         'load_2': 0,
#         'load_3': 0,
#     }
#     for i in obj:
#         if i == 5.5:
#             count_dict['load_1'] += 1
#         elif i == 8.88:
#             count_dict['load_2'] += 1
#         elif i == 11.1:
#             count_dict['load_3'] += 1
#         else:
#             print('Check load labels')
#     return count_dict
# fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (25, 7))

# #Train
# sns.barplot(data=
#            pd.DataFrame.from_dict([get_load_distribution(y_train)]).melt(),
#            x = 'variable',
#            y = 'value',
#            hue = 'variable',
#            ax = axes[0]).set_title('Class distribution in train set')

# #Validation
# sns.barplot(data=
#            pd.DataFrame.from_dict([get_load_distribution(y_val)]).melt(),
#            x = 'variable',
#            y = 'value',
#            hue = 'variable',
#            ax = axes[1]).set_title('Class distribution in validation set')

# #Test
# sns.barplot(data=
#            pd.DataFrame.from_dict([get_load_distribution(y_test)]).melt(),
#            x = 'variable',
#            y = 'value',
#            hue = 'variable',
#            ax = axes[2]).set_title('Class distribution in test set')


# In[138]:


y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)


# ### Setup the Neural Network

# In[139]:


#Reproducible
torch.manual_seed(1)


# ### Check whether the GPU is available

# In[140]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[141]:


class RegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


# In[142]:


train_dataset = RegressionDataset(torch.from_numpy(X_train).float(),
                                 torch.from_numpy(y_train).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(),
                                torch.from_numpy(y_test).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(),
                               torch.from_numpy(y_val).float())


# ### Build the feedforward neural network

# In[143]:


class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)

        return x

    def predict(self, test_inputs):
        x = self.relu(self.fc1(test_inputs))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)

        return x




# In[144]:


EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_FEATURES = len(x.columns)


# In[145]:


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


# In[146]:


model = MultipleRegression(NUM_FEATURES)
model.to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)


# In[147]:


# Define a dictionary which will store the loss / epoch for both train and validation sets
loss_stats = {
    'train':[],
    'val':[]
}


# ### Train the network

# In[150]:


print('=========Begin Training=========')
for e in tqdm(range(1, EPOCHS+1)):

    #TRAINING
    train_epoch_loss = 0
    model.train()

    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()

    #VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()

    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))

    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')




# ### Post processing
# 1. Vsiualize loss and accuracy
# 2. Test trained model

# In[151]:


train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
plt.figure(figsize=(15,8))
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')


# In[152]:


y_pred_list = []

with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())


# In[153]:


y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


# In[154]:


mse = mean_squared_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)

print("Mean Squared Error :",mse)
print("R^2 :",r_square)


# In[ ]:
