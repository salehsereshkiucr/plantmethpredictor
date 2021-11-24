import profile_generator as pg
import tensorflow as tf
import numpy as np
import preprocess as preprocess
import random
import configs
from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.constraints import max_norm


def run_experiment_cpgenie(config_list, context_list, window_size, data_size, coverage_threshold=10, include_annot=True):
    res = []
    for cnfg in config_list:
        organism_name = cnfg['organism_name']
        for context in context_list:
            sequences_onehot, methylations_train, methylations_test, annot_seqs_onehot = pg.get_processed_data(cnfg)
            PROFILE_ROWS = window_size
            PROFILE_COLS = 4
            if include_annot:
                PROFILE_COLS = 4 + 2*len(annot_seqs_onehot)
            W_maxnorm = 3
            model = Sequential()
            model.add(Conv2D(128, kernel_size=(1, 5), activation='relu', input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='same', kernel_constraint=max_norm(W_maxnorm)))
            model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
            model.add(Conv2D(256, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
            model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
            model.add(Conv2D(512, kernel_size=(1, 5), activation='relu', padding='same', kernel_constraint=max_norm(W_maxnorm)))
            model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 3)))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2))
            model.add(Activation('softmax'))
            myoptimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
            model.compile(loss='binary_crossentropy', optimizer=myoptimizer,metrics=['accuracy'])
            methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, context, window_size, coverage_threshold)
            data_size = min(data_size, 2*len(methylated_train), 2*len(unmethylated_train))
            sample_set = methylated_train[:int(data_size/2)]+unmethylated_train[:int(data_size/2)]
            random.shuffle(sample_set)
            profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, window_size=window_size)
            X, Y = pg.data_preprocess(profiles, targets, include_annot=include_annot)
            X = np.swapaxes(X, 1, 2)
            b = np.zeros((Y.size, Y.max()+1))
            b[np.arange(Y.size), Y] = 1
            Y = b
            x_train, x_val, y_train, y_val = pg.split_data(X, Y, pcnt=0.1)
            with tf.device('/device:GPU:0'):
                model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
            x_test, y_test = pg.test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, context, window_size, coverage_threshold, include_annot=include_annot)
            x_test = np.swapaxes(x_test, 1, 2)
            b = np.zeros((y_test.size, y_test.max()+1))
            b[np.arange(y_test.size), y_test] = 1
            y_test = b
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            annot_mode = 'seq-only'
            if include_annot:
                annot_mode='seq-annot'
            step_res = [organism_name, context, annot_mode, window_size, len(x_test), accuracy_score(y_test, y_pred.round()),
                                    f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
            print(step_res)
            res.append(step_res)
            np.savetxt("GFG_cpgenie.csv", res, delimiter=", ", fmt='% s')

window_size = 1600
data_size = 600000
contexts = ['CG', 'CHG', 'CHH']
coverage_threshold = 10
include_annot = False
run_experiment_cpgenie([configs.Arabidopsis_config], contexts, window_size, data_size, coverage_threshold=10, include_annot=False)
