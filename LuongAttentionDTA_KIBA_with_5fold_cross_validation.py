"""
----> LuongAttentionDTA <----
dataset: KIBA
5-fold cross validaion: yes
layer after attention: flatten() or maxpool1
"""

#%% needed packages

#pip install PyTDC
#pip install lifelines

#%% load datasets

print('>>>> dataset download <<<<')

# KIBA
from tdc.multi_pred import DTI
dataKIBA = DTI(name = 'KIBA')
splitKIBA = dataKIBA.get_split()

print('>>>> end dataset download <<<<')

#%% pre-processamento

print('>>>> preprocessing <<<<')

#join train and test to later divide in the cross-validation
import pandas as pd
dfK=splitKIBA
tudoTreino=dfK['train']
tudoTeste=dfK['test']
tudoValidation=dfK['valid']
todos=pd.concat([tudoTreino,tudoTeste,tudoValidation], axis=0)

X_Drug=todos['Drug']
X_Prot=todos['Target']
Y=todos['Y']


# protein keys
CP = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5,
               "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
               "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17,
               "W": 18, "X": 19, "Y": 20}

# drug keys
CD = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}



# label encoding + zero padding where necessary
# drugs
import numpy as np
max_drug=100
listaD=[]
for c in X_Drug:
    nova=[]
    count=0
    for v in c:
        nova.append( CD[v]  )
        count+=1
        if count==max_drug:
            break
    if len(nova) < max_drug:
        nova= nova + [0] * (max_drug-len(nova))
    listaD.append(nova)
X_Drug=np.asarray(listaD)


#proteins
max_prot=1200
listaP=[]
for c in X_Prot:
    nova=[]
    count=0
    for v in c:
        nova.append( CP[v] )
        count+=1
        if count==max_prot:
            break
    if len(nova) < max_prot:
        nova= nova + [0] * (max_prot-len(nova))
    listaP.append(nova)
X_Prot=np.asarray(listaP)

# convert targets to numpy array
Y=np.asarray(Y)


print('>>>> end of preprocessing <<<<')


#%% cross-validation
from sklearn.model_selection import KFold
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True) # define the K-fold Cross Validator

#cross validation dvides into 80/20 (train/test) by default
loss_per_fold=[]
attention_scores_per_fold=[]
mse_per_fold=[]
ci_per_fold=[]
auprc_per_fold=[]

fold_no = 1
for train, test in kfold.split(X_Drug, Y):
    # supress non essencial prints
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    tf.autograph.experimental.do_not_convert
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


    # model
    from tensorflow.keras.layers import Input, Conv1D, Dense, concatenate, Embedding, Attention, LeakyReLU, Dropout,Flatten,MaxPool1D
    from tensorflow.keras.models import Model

    inDrug = Input(shape=(100,), dtype='int32') #drug
    inProt = Input(shape=(1200,), dtype='int32') #protein

    # first branch (drugs)
    x = Embedding(input_dim=65, output_dim=128, input_length=100)(inDrug)
    x = Conv1D( filters=32,kernel_size=4)(x)
    x = Conv1D( filters=64,kernel_size=6)(x)
    x = Conv1D( filters=96,kernel_size=8)(x)
    x = Model(inputs=inDrug, outputs=x)

    #  second branch (proteins)
    y = Embedding(input_dim=65, output_dim=128, input_length=1200)(inProt)
    y = Conv1D( filters=32,kernel_size=4)(y)
    y = Conv1D( filters=64,kernel_size=6)(y)
    y = Conv1D( filters=96,kernel_size=12)(y)

    y = Model(inputs=inProt, outputs=y)

    z = Attention()([x.output,y.output])

    z=Flatten()(z)
    #maxpool1
    #z=MaxPool1D(pool_size=85)(z)

    z = Dense(1024, activation=LeakyReLU(alpha=0.01))(z)
    z = Dropout(0.5)(z)
    z = Dense(512, activation=LeakyReLU(alpha=0.01))(z)
    z = Dropout(0.5)(z)
    z = Dense(512, activation=LeakyReLU(alpha=0.01))(z)
    z = Dense(1, activation=LeakyReLU(alpha=0.01))(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    model.summary()


    #treinar o modelo
    from tensorflow.keras.optimizers import Adam
    opt = Adam(learning_rate=0.0001)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


    # train the model
    print("[INFO] training model...")
    model.fit(x=[X_Drug, X_Prot], y=Y, epochs=20, batch_size=64)

    scores = model.evaluate([X_Drug[test], X_Prot[test]], Y[test], verbose=0)

    print('loss per fold: ',scores)
    loss_per_fold.append(scores)

    # metrics
    # make predictions on the testing data
    print("[INFO] predicting binding affinities...")
    preds = model.predict([X_Drug[test], X_Prot[test]])

    y_true=np.squeeze(np.asarray(Y[test]))
    y_pred=np.squeeze(np.asarray(preds))

    #mean squared error
    from tensorflow.keras.metrics import mean_squared_error,MeanSquaredError
    mse=mean_squared_error(y_true,y_pred)
    mse_per_fold.append(mse)

    #concordance index
    import pandas as pd
    from lifelines.utils import concordance_index
    dfCI = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
    ci=concordance_index(y_true, y_pred)
    ci_per_fold.append(ci)

    #AUPR
    #convert to binary classes
    y_true_bin=np.where(y_true>=12.1, 1, y_true)
    y_true_bin=np.where(y_true<12.1, 0, y_true_bin).astype(int)

    y_pred_bin=np.where(y_pred>=12.1, 1, y_pred)
    y_pred_bin=np.where(y_pred<12.1, 0, y_pred_bin).astype(int)

    import sklearn.metrics
    auprc = sklearn.metrics.average_precision_score(y_true_bin, y_pred_bin)
    auprc_per_fold.append(auprc)

    # increase fold number
    fold_no = fold_no + 1

#%% results of the 5-fold cross-validation

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> MSE: {np.around(np.mean(mse_per_fold),2)} (+- {round(np.std(mse_per_fold),2)})')
print(f'> CI: {np.around(np.mean(ci_per_fold),2)} (+- {round(np.std(ci_per_fold),2)})')
print(f'> AUPRC: {np.around(np.mean(auprc_per_fold),2)} (+- {round(np.std(auprc_per_fold),2)})')
print('------------------------------------------------------------------------')
