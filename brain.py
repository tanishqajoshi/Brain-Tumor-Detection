#!/usr/bin/env python
# coding: utf-8

# In[22]:


#pip install tk


# # Importing the modules 

# In[23]:


import tkinter
from tkinter.filedialog import askopenfilename


# In[24]:


#pip install filetype


# In[25]:


import filetype
import os
import keras


# In[26]:


#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# In[27]:


import livelossplot
plot_losses = livelossplot.PlotLossesKeras()


# In[28]:


from keras.models import Sequential


# In[29]:


from keras.models import Model, load_model


# In[30]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, ZeroPadding2D, Activation
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[31]:


plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 


# # OneHotEncoder
# 

# In[32]:


encoder = OneHotEncoder()
encoder.fit([[0], [1]])


# # Creating Empty Lists

# In[33]:


data = []
paths = []
result = []


# # Walking through the Dataset via import os

# In[34]:


os.listdir(r'C:\Users\hp\Desktop\Brain Tumor Detection')


# In[35]:


for r, d, f in os.walk(r'C:\Users\hp\Desktop\Brain Tumor Detection\yes'):
    for file in f:
        if '.JPG' in file:
            paths.append(os.path.join(r, file))


# In[36]:


paths


# In[37]:


for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())


# In[38]:


paths = []
for r, d, f in os.walk(r'C:\Users\hp\Desktop\Brain Tumor Detection\no'):
    for file in f:
        if '.JPG' in file:
            paths.append(os.path.join(r, file))


# In[39]:


paths


# In[40]:


for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())


# In[41]:


data.shape


# In[42]:


data = np.array(data)
data.shape


# In[43]:


result = np.array(result)
result = result.reshape(74,2)
result.shape


# In[44]:


x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=42)


# # Build Model

# In[45]:


def build_model(input_shape):
    
     
    X_input = Input(input_shape) 
   
    X = ZeroPadding2D((2, 2))(X_input) 
    
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    
   
    X = MaxPooling2D((4, 4), name='max_pool0')(X) 
    
    X = MaxPooling2D((4, 4), name='max_pool1')(X) 
    
    
    X = Flatten()(X) 
    
    X = Dense(2, activation='sigmoid', name='fc')(X) 
    
    
    model = Model(inputs = X_input, outputs = X, name='BrainTumourDetectionModel')
    
    return model


# In[46]:


img_shape = (128,128,3)


# In[47]:


model = build_model(img_shape)


# # Model Summary

# In[48]:


model.summary


# In[49]:


model.compile(loss = "binary_crossentropy", metrics = ['accuracy'] , optimizer='adam')
print(model.summary())


# In[50]:


y_train.shape


# In[51]:


history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, callbacks = [plot_losses], validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test losses : ", score[0])
print("Test acc : ", score[1])


# In[53]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test Loss', 'Validation Loss'], loc='upper right')
plt.show()


# In[54]:


#pip install Image


# In[92]:


from matplotlib.pyplot import imshow


# In[93]:


root = tkinter.Tk()
path = askopenfilename(initialdir = root, title = "Select an image", filetypes = [("jpg files",".jpg"), ("jpeg files", ".jpeg")]) 
root.destroy()
if len(path) != 0:
    if not(filetype.is_image(path)):
        print("Selected file is not an image")
        exit()
else:
    print("No file Selected")
    exit()

img = Image.open(path)
x = np.array(img.resize((128,128)))


# In[94]:


x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
text = (str(res[0][classification]*100) + '\n % Confidence This Is Not A Brain Tumor ' )
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial",size= 14)
pdf.cell(180,20,txt="Tumor Report",ln=1,align="C")
pdf.cell(40,10,f'It is not a brain tumor {text}!')
pdf.output('test.pdf')


# In[95]:


from matplotlib.pyplot import imshow


# In[105]:


root = tkinter.Tk()
path = askopenfilename(initialdir = root, title = "Select an image", filetypes = [("jpg files",".jpg"), ("jpeg files", ".jpeg")]) 
root.destroy()
    
if len(path) != 0:
    if not(filetype.is_image(path)):
        print("Selected file is not an image")
        exit()
else:
    print("No file Selected")
    exit()

img = Image.open(path)


# In[106]:


x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)

text= (str(res[0][classification]*100) + '% Confidence This Is A Tumor''\n' )
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial",size= 14)
pdf.cell(180,20,txt="Tumor Report",ln=1,align="C")
pdf.cell(40,10,f' It is a brain tumor {text} !')
pdf.output('add.pdf')


# In[ ]:





# In[ ]:




