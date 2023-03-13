# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
      for j in range(r_u[0]):
        if lab_arr[i,0] == lab_arr_unique[j]:
          one_hot_enc[i,j] = 1
    
    return one_hot_enc

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and standard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, UpSampling1D
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose, Concatenate, Multiply, Add, MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose, MaxPooling3D, GlobalAveragePooling2D, Conv1DTranspose, UpSampling2D
from keras import Model
from keras.layers import BatchNormalization,Conv1D, Activation, Layer, MaxPooling1D, GRU, Bidirectional
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from sklearn.metrics import confusion_matrix
import os

def get_gt_index(y):

  import tqdm

  shapeY = np.shape(y)

  pp,qq = np.unique(y, return_counts=True)
  sum1 = np.sum(qq)-qq[0]

  index = np.empty([sum1,3], dtype = 'int')

  cou = 0
  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeY[0]):
      for j in range(shapeY[1]):
        if y[i,j] == k:
          index[cou,:] = np.expand_dims(np.array([k,i,j]),0)
          #print(cou)
          cou = cou+1
  return index

# The code takes the entire hsi/lidar image as input for 'X' and groundtruth file as input for 'y'
# and the patchsize as for 'windowSize'.
# The output are the patches centered around the groundtruth pixel, the corresponding groundtruth label and the
# pixel location of the patch.

def make_patches(X, y, windowSize):

  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.zeros([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = get_gt_index(y)
  
  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]], dtype = 'float32')
  patchesY = np.empty([index.shape[0]],dtype = 'uint8')

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    patchesY[i] = index[i,0]

  return patchesX, patchesY, index

# Read train and test samples

train_vec = np.reshape(np.load('.data/train_patches.npy')[:,5,5,:], [-1,1,200,1])
train_labels = np.load('.data/train_labels.npy')

test_vec = np.reshape(np.load('.data/test_patches.npy')[:,5,5,:], [-1,1,200,1])
test_labels = np.load('.data/test_labels.npy')

# 1D Feedback convolution based encoder block

def fb_encoder_block(x):

  c0 = Conv2D(64,(1,7), activation="relu", strides=(1,2), padding="valid")(x)
  cL1 = Conv2D(64,(1,7), activation="relu", strides=1, padding="same")
  cL2 = Conv2D(64,(1,7), activation="relu", strides=1, padding="same")
  cL3 = Conv2D(64,(1,7), activation="relu", strides=1, padding="same")

  ## Satge I
  
  c1 = cL1(c0)
  #c1 = BatchNormalization()(c1)
  ad1 = Add()([c0, c1])
  c2 = cL2(ad1)
  #c2 = BatchNormalization()(c2)
  ad2 = Add()([c0, c1, c2])
  c3 = cL3(ad2)

  ## Satge II

  ad3 = Add()([c2,c3])
  c1 = cL1(ad3)
  #c1 = BatchNormalization()(c1)
  ad4 = Add()([c1,c3])
  c2 = cL2(ad4)
  #c2 = BatchNormalization()(c2)
  ad5 = Add()([c1,c2])
  c3 = cL3(ad5)
  cp = MaxPooling2D(pool_size = (1,2), strides = (1,2))(c3)

  return cp

# 1D CNN based decoder block
def decoder_block(x):

  c0 = Conv2DTranspose(32,(1,7), activation="relu", strides=(1,2), padding="valid")(x)
  cL1 = Conv2D(32,(1,7), activation="relu", strides=1, padding="same")
  cL2 = Conv2D(32,(1,7), activation="relu", strides=1, padding="same")
  cL3 = Conv2D(32,(1,7), activation="relu", strides=1, padding="same")

  ## Satge I
  
  c1 = cL1(c0)
  #c1 = BatchNormalization()(c1)
  ad1 = Add()([c0, c1])
  c2 = cL2(ad1)
  #c2 = BatchNormalization()(c2)
  ad2 = Add()([c0, c1, c2])
  c3 = cL3(ad2)
  #c3 = BatchNormalization()(c3)

  ## Satge II

  #ad3 = Add()([c2,c3])
  #c1 = cL1(ad3)
  #c1 = BatchNormalization()(c1)
  #ad4 = Add()([c1,c3])
  #c2 = cL2(ad4)
  #c2 = BatchNormalization()(c2)
  #ad5 = Add()([c1,c2])
  #c3 = cL3(ad5)
  cp = UpSampling2D(size = (1,2))(c3)

  return cp

xA = Input(shape = (1,200,1))

# Encoder
b1 = fb_encoder_block(xA)
b2 = fb_encoder_block(b1)
b3 = fb_encoder_block(b2)
flt = Flatten()(b3)

# Encoder representation
cd1 = Dense(10, activation = 'relu')(flt)
cd2 = Dense(192, activation = 'relu')(cd1)
rshp1 = Reshape([1,1,192])(cd2)

# Decoder

b4 = decoder_block(rshp1)
b5 = decoder_block(b4)
flt2 = Flatten()(b5)
cd3 = Dense(200, activation = 'relu')(flt2)
rshp2 = Reshape([1,200,1])(cd3)
decoder = Model(xA, rshp2, name="decoder")
decoder.compile(loss = 'mean_squared_error', optimizer=keras.optimizers.Nadam(0.0005), metrics = ['mse'])
decoder.summary()

keras.utils.plot_model(decoder)

# Random Forest Classifier to test the performance of the encoded representation
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=200)

# Training the model

acc_temp=0 # temporary accuracy to compare with max accuracy
import gc
for epoch in range(500): 
  gc.collect()
  keras.backend.clear_session()
  decoder.fit(x = train_vec, y = train_vec,
                  epochs=1, batch_size = 64, verbose = 1)

  new_model = Model(decoder.input, decoder.layers[32].output, name = 'new_model') 
  code_feat_train = np.reshape(new_model.predict(train_vec),[1024,10])
  code_feat_test = np.reshape(new_model.predict(test_vec),[9225,10])

  clf.fit(code_feat_train, train_labels)
  preds = clf.predict(code_feat_test)
  del code_feat_train, code_feat_test
  conf = confusion_matrix(test_labels, preds)
  ovr_acc, _, _, _, _ = accuracies(conf)

  print(epoch)
  print(np.round(100*ovr_acc,2))
  if ovr_acc>=acc_temp:
    decoder.save('.models/model')
  print('acc_max = ', np.round(100*acc_temp,2), '% at epoch', ep)
