import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import Activation,Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import configs as configs



def run_experiment(organism_name, context, mode, root, i):
    test_percent = 0.2
    test_val_percent = 0.5

    X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
    Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle= True)

    Y = np.asarray(pd.cut(Y, bins = 2, labels=[0,1], right=False))
    X = X.reshape(list(X.shape) + [1])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)

    PROFILE_COLS = X.shape[2]
    PROFILE_ROWS = X.shape[1]

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(1, PROFILE_COLS), activation='relu', input_shape=(PROFILE_ROWS, PROFILE_COLS,1)))
    model.add(Reshape((10, 10,16), input_shape=(100,1, 16)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    with tf.device('/device:GPU:0'):
        model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))

    y_pred = model.predict(x_test)
    step_res = [organism_name, context, mode, str(i), accuracy_score(y_test, y_pred.round()), f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
    print(step_res)

    return step_res


root = '/home/csgrads/ssere004/output/'
contexts = ['CG', 'CHG', 'CHH']

cnfgs = [configs.Cucumber_config, configs.Tomato_config, configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config]

res = []
for cnfg in cnfgs:
    for context in contexts:
        for i in range(1, 5):
            organism_name = cnfg['organism_name']
            root = '/home/csgrads/ssere004/Organisms/'+ cnfg['organism_name'] +'/profiles/'
            anno_types = cnfg['annot_types']
            for at in anno_types:
                mode = 'Seq' + at
                res.append(run_experiment(organism_name, context, mode, root, i))
                np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
            mode = 'SeqOnly'
            res.append(run_experiment(organism_name, context, mode, root, i))
            np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
