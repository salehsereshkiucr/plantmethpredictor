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


window_sizes = [100, 200, 400, 800, 1000]
block_sizes = [(10, 10), (20, 10), (20, 20), (40, 20), (50, 20)]

contexts = ['CG', 'CHG', 'CHH']

test_percent = 0.2
test_val_percent = 0.5

root = '/home/csgrads/ssere004/output_cpgenieannot/'

def data_preprocess(X, Y, window_size):
    X = np.delete(X, range(4, X.shape[2]), 2)
    b = [j for j in range(int((1000-window_size)/2))] + [j for j in range(1000 - int((1000-window_size)/2), 1000)]
    X = np.delete(X, b, 1)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)
    return x_train, y_train, x_test, y_test, x_val, y_val

def model(PROFILE_ROWS, PROFILE_COLS, block_size):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 1), input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='same', use_bias=True))
    model.add(Reshape((block_size[0], block_size[1], 16), input_shape=(PROFILE_ROWS, 1, 16))) #end of first
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  padding='same', use_bias=True))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')) #end of second
    model.add(Conv2D(48, kernel_size=(3, 3), padding='VALID', use_bias=True))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID', use_bias=True)) #end of third
    model.add(Reshape((-1, 2*2*64), input_shape=(2, 2, 64)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

res = []
for context in contexts:
    for i in range(1, 4):
        for w in range(len(window_sizes)):
            window_size = window_sizes[w]
            X = np.load(root + organism_name + '/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
            Y = np.load(root + organism_name + '/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)
            x_train, y_train, x_test, y_test, x_val, y_val = data_preprocess(X, Y, window_size)
            model = model(x_train.shape[1], x_train.shape[2], block_sizes[w])
            opt = tf.keras.optimizers.SGD(lr=0.01)
            model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
            with tf.device('/device:GPU:0'):
                model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
            y_pred = model.predict(x_test)
            step_res = [organism_name, context, 'seq-only', window_size, str(i), accuracy_score(y_test, y_pred.round()), f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
            print(step_res)
            res.append(step_res)
            np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
