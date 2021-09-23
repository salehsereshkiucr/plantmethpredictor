from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.constraints import max_norm
import numpy as np
from sklearn.model_selection import train_test_split
import configs as configs
import pandas as pd


def run_experiment(organism_name, context, i, root, mode):
    X = np.load(root + organism_name +'/profiles/' + str(i) + '/X_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
    Y = np.load(root + organism_name +'/profiles/' + str(i) + '/Y_' + context + '_' + mode + '_' + organism_name + '.npy', allow_pickle=True)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size),Y] = 1
    Y = b
    X = X.reshape(list(X.shape) + [1])
    X = np.swapaxes(X,1,2)

    PROFILE_COLS = X.shape[1]
    PROFILE_ROWS = X.shape[2]

    test_percent = 0.2
    test_val_percent = 0.5
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)

    W_maxnorm = 3

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(1, 5), activation='relu', input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1,3)))
    model.add(Conv2D(256, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1,3)))
    model.add(Conv2D(512, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1,3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    myoptimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='binary_crossentropy', optimizer=myoptimizer,metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=0, validation_split=0.1)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    step_res = [organism_name, context, 'cpgenie', str(i), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred)]
    print(step_res)
    return step_res


root = '/home/csgrads/ssere004/output_cpgenie/'
contexts = ['CG', 'CHG', 'CHH']
res=[]



#residues = [('CHH', 1), ('CHG', 4)]
#for context, i in residues:
#    res.append(run_experiment(organism_name, context, i, root))
#    np.savetxt("GFG_cpgenie.csv", res, delimiter=", ", fmt='% s')

cnfgs = [configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
for cnfg in cnfgs:
    organism_name = cnfg['organism_name']
    for i in range(1, 2):
        for context in contexts:
            for mode in cnfg['annot_types']:
                res.append(run_experiment(organism_name, context, i, root, mode))
            np.savetxt("GFG_cpgenie.csv", res, delimiter=", ", fmt='% s')

