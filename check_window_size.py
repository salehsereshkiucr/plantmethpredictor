import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import configs as configs

config = configs.Arabidopsis_config
organism_name = config['organism_name']
mode = 'Seq'
for at in config['annot_types']:
    mode += at


window_sizes = [100, 200, 400, 800, 1600, 3200]
block_sizes = [(10, 10), (20, 10), (20, 20), (40, 20), (40, 40), (80, 40)]

contexts = ['CG', 'CHG', 'CHH']

test_percent = 0.2
test_val_percent = 0.5

root = '/home/csgrads/ssere004/'

def data_preprocess(X, Y, window_size):
    X = np.delete(X, range(4, X.shape[2]), 2)
    b = [j for j in range(int((3200-window_size)/2))] + [j for j in range(3200 - int((3200-window_size)/2), 3200)]
    X = np.delete(X, b, 1)
    X = X.reshape(list(X.shape) + [1])
    #X = np.swapaxes(X, 1, 2)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)
    return x_train, y_train, x_test, y_test, x_val, y_val

def model(PROFILE_ROWS, PROFILE_COLS, block_size):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, PROFILE_COLS), activation='relu', input_shape=(PROFILE_ROWS, PROFILE_COLS, 1)))
    model.add(Reshape((block_size[0], block_size[1], 16), input_shape=(PROFILE_ROWS, 1, 16)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

res = []
for context in contexts:
    for i in range(1, 2):
        for w in range(len(window_sizes)):
            print(context, i, w)
            window_size = window_sizes[w]
            X = np.load(root + organism_name + '/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
            Y = np.load(root + organism_name + '/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)
            x_train, y_train, x_test, y_test, x_val, y_val = data_preprocess(X, Y, window_size)
            print('data preprocessed')
            print('model inputs', x_train.shape[1], x_train.shape[2], block_sizes[w])

            PROFILE_ROWS = x_train.shape[1]
            PROFILE_COLS = x_train.shape[2]
            block_size = block_sizes[w]


            model = Sequential()
            model.add(Conv2D(16, kernel_size=(1, PROFILE_COLS), activation='relu', input_shape=(PROFILE_ROWS, PROFILE_COLS, 1)))
            model.add(Reshape((block_size[0], block_size[1], 16), input_shape=(PROFILE_ROWS, 1, 16)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))


            print('model processed')
            opt = tf.keras.optimizers.SGD(lr=0.01)
            model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
            print('ready to fit')
            with tf.device('/device:GPU:0'):
                model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
            y_pred = model.predict(x_test)
            step_res = [organism_name, context, 'seq-only', window_size, str(i), accuracy_score(y_test, y_pred.round()), f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
            print(step_res)
            res.append(step_res)
            np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')



#_______________________________________________________________
