#!/usr/bin/env python
# coding: utf-8

# In[184]:


import numpy as np
import pandas as pd 
import tensorflow as tf
import glob 
import matplotlib.pyplot as plt
import keras


# In[185]:


train_files=glob.glob("C:/Users/Computer Maestro/Downloads/chest_xray/train/*/**")


# In[186]:


train_files


# In[187]:


from random import shuffle
shuffle(train_files)


# In[188]:


train_files


# In[189]:


from PIL import Image
im = Image.open(train_files[591])
im_arr=np.array(im)
plt.imshow(im);


# In[275]:


#v=im_arr.shape
train_files[7][53]


# In[191]:


#v=(128,128)


# In[192]:


len(train_files)


# In[193]:


# count=0
# for i,trf in enumerate(train_files):
#     im=Image.open(trf)
#     im_arr=np.array(im)
#     if(im_arr.shape==v):
#         count=count+1
    


# In[194]:


#count


# In[195]:


import cv2


# In[246]:


img = cv2.imread(train_files[0])


# In[247]:


img=Image.open(train_files[0])


# In[248]:


imar=np.array(im)
print(imar)


# In[249]:


#plt.imshow(img)


# In[250]:


im1= cv2.resize(imar, v, interpolation = cv2.INTER_AREA)


# In[251]:


im.resize((128,128))


# In[252]:


#im


# In[311]:


import re
def index(s,ch):
    a=[]
    for m in re.finditer(ch,s):
        a.append(m.start())
    return a


# In[312]:


X_train=np.zeros((len(train_files),128,128))
Y_train=np.zeros((len(train_files),1))
for i, fi in enumerate(train_files):
    img2=Image.open(fi)
    np.array(im).resize
    im=im.resize((128,128))
    X_train[i,:,:]=np.array(im)
    Y_train[i]=fi[-6]


# In[313]:


for i, trf in enumerate(train_files):
    idx=index(trf,'_')
    if(trf[idx[-1]+1=='b']):
        tk=[trf,1]
    else:
        tk=[trf,2]
        train_files.append(tk)


# In[314]:


X_train[0]


# In[315]:


train_files


# In[316]:


#for i, trf in enumerate(train_files):
 #   im = cv2.imread(trf)
 #   imconv=cv2.resize(im, (128,128), interpolation = cv2.INTER_AREA)
  #  X_train[i,:,:,:]=imconv


# In[317]:


X_train/255.0


# In[318]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


# In[319]:


model= keras.Sequential([
    # 1
    keras.layers.Conv2D(
    filters=120,
    kernel_size=3,
        activation='relu',
    input_shape=(128,128,1)
    ),
    
    keras.layers.Conv2D(
    filters=24,
    kernel_size=3,
        activation='relu',
    ),
  
    keras.layers.Flatten(),

    keras.layers.Dense(
        units=96,
        activation='relu',
    ),
 
    keras.layers.Dense(3,activation='softmax')
    
])


# In[320]:


model=keras.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),strides=2,padding='Same',activation='relu',input_shape
=(128, 128, 1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3)) #here added dropout model.add(keras.layers.Conv2D(64,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256,(3,3), strides=1,padding= 'Same', activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(3, activation = "softmax"))



# In[321]:


model.compile(optimizer=keras.optimizers.Adam(learning_rate= 1e-3),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


# In[322]:


model.summary()


# In[323]:


X_train=X_train.reshape(len(X_train),128,128,1)
Y_train.shape


# In[324]:


X_train.shape


# In[325]:


model.fit(X_train, Y_train,epochs=2,validation_split=0.1)


# In[ ]:




