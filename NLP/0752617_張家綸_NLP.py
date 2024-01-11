#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import jieba
from gensim.models import word2vec, fasttext
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.layers.embeddings import Embedding
from tensorflow.keras.losses import categorical_crossentropy
from keras.layers import *


# In[2]:


train_text = pd.read_csv('train_data.csv')
test_text = pd.read_csv('test_data.csv')
total_text = pd.concat([train_text['title'], test_text['title']], axis=0)
total_key = pd.concat([train_text['keyword'], test_text['keyword']], axis=0)
total_text = total_text.reset_index(drop=True)
total_key = total_key.reset_index(drop=True)


# In[5]:


total_keydrop = total_key.dropna(axis=0)  
keyword = []
for i in total_keydrop.index:
    if i%5000 == 0:
        print(i)
    keyword = keyword + total_key[i].split(',')
##userDict.txt
file = open('userDict.txt','w', encoding="utf-8")
file.writelines(["%s\n" % item for item in keyword])
file.close()


# In[6]:


# 中文停用詞
stopwords = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]
# 自定義中文詞彙
jieba.load_userdict('userDict.txt')
# 定義中文分詞，包含去除掉停用詞
def title_preprocessing(train_text):
    all_words = list(jieba.cut(train_text, cut_all=False))
    fileTrainSeg = []
    for word in all_words:
        if word not in stopwords:
            fileTrainSeg.append(word)
    return(fileTrainSeg)


# In[18]:


total_keydot = total_key.fillna(',')
corpus= total_text.astype('str').apply(title_preprocessing)
corpuskey = total_keydot.astype('str').apply(title_preprocessing)
corpus.shape


# In[52]:


newcorpus = corpus + corpuskey


# In[19]:


import keras
#詞庫有多少詞彙
MAX_NUM_WORDS = 100000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)


# In[54]:


# 詞集&轉成vector
tokenizer.fit_on_texts(newcorpus)
X_total = tokenizer.texts_to_sequences(newcorpus)


# In[56]:


# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 30
X_total = keras.preprocessing.sequence.pad_sequences(X_total, maxlen=MAX_SEQUENCE_LENGTH)
y_train = keras.utils.to_categorical(train_text['label'])


# In[57]:


X_train = X_total[:len(train_text)]
X_test = X_total[len(train_text):]


# In[58]:


# from sklearn.model_selection import StratifiedKFold
# num_folds = 2
# kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


# In[65]:


optimizer = Adam()
loss_function = categorical_crossentropy
#for train, test in kfold.split(X_train, train_text['label']):
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 128

# 一個詞向量的維度
NUM_EMBEDDING_DIM = 128

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 64

# 是否顯示進度條
verbosity = 1

# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import Input
from keras.layers import Embedding, LSTM, concatenate, Dense
from keras.models import Model

# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
train_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
# 為 64
embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
train_embedded = embedding_layer(train_input)

# LSTM 層
# 兩個新聞標題經過此層後
# 為一個 64 維度向量
shared_lstm1 = LSTM(NUM_LSTM_UNITS, return_sequences = True, dropout=0.2)
train_lstm1 = shared_lstm1(train_embedded)
shared_lstm2 = LSTM(NUM_LSTM_UNITS, dropout=0.2)
train_lstm2 = shared_lstm2(train_lstm1)

# 全連接層搭配 Softmax Activation
# 可以回傳 10 個成對標題
# 屬於各類別的可能機率
dense =  Dense(units=NUM_CLASSES, activation='softmax')
predictions = dense(train_lstm2)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(inputs=train_input,  outputs=predictions)
# Loss function
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Fit data to model
history = model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          verbose=verbosity,
          shuffle=True)


# In[60]:


# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#     print('------------------------------------------------------------------------')
#     print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')


# In[61]:


predictions  = model.predict(X_test)


# In[62]:


predictions.argmax(axis=1)
pred_test = pd.concat([test_text['id'], pd.DataFrame(predictions.argmax(axis=1))], axis=1)


# In[63]:


pred_test.columns = ['id', 'label']
pred_test.to_csv('predv1.csv', index=None)

