"""
----> Authors <-----
Carolina Fernandes ()
Denis Rocha (2017278968)

----> LuongAttentionDTA <----
dataset: DAVIS
5-fold cross validaion: no
layer after attention: flatten() or maxpool2
"""

#%% needed packages

#pip install PyTDC
#pip install lifelines

#%% load datasets

print('>>>> dataset download <<<<')

# Davis
from tdc.multi_pred import DTI
dataDavis = DTI(name = 'DAVIS')
splitDavis = dataDavis.get_split()


print('>>>> end of dataset download <<<<')

#%% preprocessing

print('>>>> preprocessing <<<<')

import pandas as pd
dfD=splitDavis
tudoTreino=dfD['train']
tudoTeste=dfD['test']
tudoValidation=dfD['valid']
todos=pd.concat([tudoTreino,tudoTeste,tudoValidation], axis=0)

#Davis size: 25772
n=25772
tamanho=round(n/1) #use all

X_Drug=todos['Drug'].iloc[0:tamanho]
X_Prot=todos['Target'].iloc[0:tamanho]
Y=todos['Y'].iloc[0:tamanho]

#apply log to binding affinities
import math
Y=Y.apply(lambda x: -math.log((x/1e9),10) )


#%%
# protein keys
CP = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5,
               "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
               "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17,
               "W": 18, "X": 19, "Y": 20}

# crugs keys
CD = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}



# label encoding + zero padding where necessary
#drugs
import numpy as np
max_drug=100
listaD=[]
for c in X_Drug: # ir buscar cada uma das strings
    nova=[]
    count=0
    for v in c: # ir buscar cada letra das strings
        nova.append( CD[v]  )
        count+=1
        if count==max_drug:
            break
    if len(nova) < max_drug:
        nova= nova + [0] * (max_drug-len(nova))
    listaD.append(nova)
X_Drug=np.asarray(listaD)#.astype(np.int32)

#proteins
max_prot=1200
listaP=[]
for c in X_Prot: # ir buscar cada uma das strings
    nova=[]
    count=0
    for v in c: # ir buscar cada letra das strings
        nova.append( CP[v] )
        count+=1
        if count==max_prot:
            break
    if len(nova) < max_prot:
        nova= nova + [0] * (max_prot-len(nova))
    listaP.append(nova)
X_Prot=np.asarray(listaP)#.astype(np.int32)

# convert targets to arrays
Y=np.asarray(Y)


print('>>>> end of preprocessing <<<<')



#%% dataset split
max_id=len(Y)
import numpy as np
indices = np.arange(max_id)
from sklearn.model_selection import train_test_split
train, test = train_test_split(indices, test_size=0.2, train_size=None) # 80/20 (train/test)

#%% model and training

# supress non essencial prints
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.autograph.experimental.do_not_convert
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# model
from tensorflow.keras.layers import Input, Conv1D, Dense, Concatenate, Embedding, Attention, LeakyReLU, Dropout,Flatten,MaxPool1D,Reshape,Permute
from tensorflow.keras.models import Model


inDrug = Input(shape=(100,),  dtype='int32') #drug
inProt = Input(shape=(1200,), dtype='int32') #protein

# first branch (drugs)
x = Embedding(input_dim=65, output_dim=128, input_length=100)(inDrug)
x = Conv1D( filters=32,kernel_size=4)(x)
x = Conv1D( filters=64,kernel_size=6)(x)
x = Conv1D( filters=96,kernel_size=8)(x)
x = Model(inputs=inDrug, outputs=x)

# second branch (proteins)
y = Embedding(input_dim=65, output_dim=128, input_length=1200)(inProt)  
y = Conv1D( filters=32,kernel_size=4)(y)
y = Conv1D( filters=64,kernel_size=6)(y)
y = Conv1D( filters=96,kernel_size=12)(y)

y = Model(inputs=inProt, outputs=y)

z = Attention()([x.output,y.output])

z=Flatten()(z)
#maxpool2
#z1=MaxPool1D(pool_size=85,data_format='channels_last')(z)
#z2=MaxPool1D(pool_size=96,data_format='channels_first')(z)
#z2=Permute((2, 1))(z2)
#z=Concatenate()([z1, z2])


z = Dense(1024, activation=LeakyReLU(alpha=0.01))(z)
z = Dropout(0.5)(z)
z = Dense(1024, activation=LeakyReLU(alpha=0.01))(z)
z = Dropout(0.5)(z)
z = Dense(512, activation=LeakyReLU(alpha=0.01))(z)
z = Dense(1, activation=LeakyReLU(alpha=0.01))(z)

model = Model(inputs=[x.input, y.input], outputs=z)

layer_outputs = [layer.output for layer in model.layers]
viz_model = Model(inputs=[x.input, y.input], outputs=layer_outputs)

model.summary()


#training
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
opt = Adam(learning_rate=0.0001)
model.compile(loss=MeanSquaredError(), optimizer=opt)
print("[INFO] training model...")
history=model.fit(x=[X_Drug, X_Prot], y=Y, epochs=350, batch_size=64)

scores = model.evaluate([X_Drug[test], X_Prot[test]], Y[test], verbose=0)

# to visualize attention scores
features = viz_model.predict([X_Drug[test], X_Prot[test]])
#attention scores -> outputs[10]
attention_scores=features[10]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure()
plt.imshow(attention_scores[1,:,:], interpolation='nearest')#, cmap=cm.Greys_r)
plt.ylabel('Drug')
plt.xlabel('Protein')
plt.title('Attention Scores')
plt.colorbar()
plt.show()



# metrics
# make predictions on the testing data
print("[INFO] predicting binding affinities...")
preds = model.predict([X_Drug[test], X_Prot[test]])

y_true=np.squeeze(np.asarray(Y[test]))
y_pred=np.squeeze(np.asarray(preds))

#mean squared error
from tensorflow.keras.metrics import mean_squared_error,MeanSquaredError
mse=mean_squared_error(y_true,y_pred)

#concordance index
import pandas as pd
from lifelines.utils import concordance_index
dfCI = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
ci=concordance_index(y_true, y_pred)

#AUPR
# convert in binary classes
y_true_bin=np.where(y_true>=7, 1, y_true)
y_true_bin=np.where(y_true<7, 0, y_true_bin).astype(int)

y_pred_bin=np.where(y_pred>=7, 1, y_pred)
y_pred_bin=np.where(y_pred<7, 0, y_pred_bin).astype(int)

import sklearn.metrics
auprc = sklearn.metrics.average_precision_score(y_true_bin, y_pred_bin)

#%% plot loss
import matplotlib.pyplot as plt
import numpy
plt.figure()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train', loc='upper left')
plt.show()

#%% print metrics
print('------------------------------------------------------------------------')
print('Scores:')
print(f'> MSE: {mse} )')
print(f'> CI: {ci} )')
print(f'> AUPRC: {auprc} )')
print('------------------------------------------------------------------------')


#%% predicted vs true binding affinities plot
plt.figure()
plt.plot(y_true,y_pred,'o', color='black')
plt.plot(np.arange(5,20),np.arange(5,20),'--', color='black')
plt.title('True vs Predicted')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.show()
