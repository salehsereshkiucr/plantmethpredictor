import configs as configs
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

def model(X_train, Y_train, X_test, Y_test):
    from hyperas.distributions import choice
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation,Flatten,Merge
    from keras.layers.convolutional import Convolution2D,MaxPooling2D
    from keras.optimizers import RMSprop
    from hyperas.distributions import choice
    from keras.callbacks import ModelCheckpoint
    from keras.constraints import maxnorm
    import tensorflow as tf
    datasize = 1000
    W_maxnorm = 3
    DROPOUT = {{choice([0.3,0.5,0.7])}}
    model = Sequential()
    model.add(Convolution2D(128, 1, 5, border_mode='same', input_shape=(4, 1, datasize),activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5),strides=(1,3)))
    model.add(Convolution2D(256,1,5, border_mode='same',activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5),strides=(1,3)))
    model.add(Convolution2D(512,1,5, border_mode='same',activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5),strides=(1,3)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    myoptimizer = RMSprop(lr={{choice([0.01,0.001,0.0001])}}, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=myoptimizer,metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=5,validation_split=0.1)
    score, acc = model.evaluate(X_test,Y_test)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def data():
    import numpy as np
    from sklearn.model_selection import train_test_split
    import configs as configs
    import pandas as pd
    root = '/home/csgrads/ssere004/output_cpgenie/'
    contexts = ['CG', 'CHG', 'CHH']
    cnfg = configs.Arabidopsis_config
    organism_name = cnfg['organism_name']
    context = contexts[0]
    i = 1
    X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '__' + organism_name + '.npy', allow_pickle=True)
    Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '__' + organism_name + '.npy', allow_pickle= True)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    X = X.reshape(list(X.shape) + [1])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=None)
    return x_train, x_test, y_train, y_test

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())

