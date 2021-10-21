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

contexts = ['CG', 'CHG', 'CHH']

window_size = 1000

test_percent = 0.2
test_val_percent = 0.5

root = '/home/csgrads/ssere004/output_cpgenieannot/'

def data_preprocess(X, Y, window_size):
    X = np.delete(X, range(4, X.shape[2]), 2)
    b = [j for j in range(int((1000-window_size)/2))] + [j for j in range(1000 - int((1000-window_size)/2), 1000)]
    X = np.delete(X, b, 1)
    X = X.reshape(list(X.shape) + [1])
    #X = np.swapaxes(X, 1, 2)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=test_percent, random_state=None)
    return x_train, y_train, x_val, y_val


res = []
for context in contexts:
    for i in range(1, 2):
        for context_test in contexts:
            X = np.load(root + organism_name + '/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
            Y = np.load(root + organism_name + '/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)

            x_test = np.load(root + organism_name + '/profiles/' + str(i) + '/X_' + context_test + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
            y_test = np.load(root + organism_name + '/profiles/' + str(i) + '/Y_' + context_test + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)

            x_train, y_train, x_val, y_val = data_preprocess(X, Y, window_size)
            print('data preprocessed')

            PROFILE_ROWS = x_train.shape[1]
            PROFILE_COLS = x_train.shape[2]
            block_size = (50, 20)


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
            mode = context+'-'+context_test
            step_res = [organism_name, context, 'seq-only', window_size, mode, str(i), accuracy_score(y_test, y_pred.round()), f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
            print(step_res)
            res.append(step_res)
            np.savetxt("GFG.csv", res, delimiter=", ", fmt ='% s')
