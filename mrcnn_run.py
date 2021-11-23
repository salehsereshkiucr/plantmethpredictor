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


def run_experiment_mrcnn(config_list, context_list, window_size, block_size, data_size, coverage_threshold=10, include_annot=True):
    res = []
    for cnfg in config_list:
        organism_name = cnfg['organism_name']
        for context in context_list:
            sequences_onehot, methylations_train, methylations_test, annot_seqs_onehot = pg.get_processed_data(cnfg)
            PROFILE_ROWS = window_size
            PROFILE_COLS = 4
            if include_annot:
                PROFILE_COLS = 4 + 2*len(annot_seqs_onehot)
            model = Sequential()
            model.add(Conv2D(16, kernel_size=(PROFILE_COLS, 1), input_shape=(PROFILE_COLS, PROFILE_ROWS, 1), padding='VALID', use_bias=True))
            model.add(Reshape((block_size[0], block_size[1], 16), input_shape=(PROFILE_ROWS, 1, 16))) #end of first
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  padding='VALID', use_bias=True))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(3,3), padding='VALID')) #end of second
            model.add(Conv2D(48, kernel_size=(3, 3), padding='VALID', use_bias=True))
            model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID', use_bias=True)) #end of third
            model.add(Reshape((-1, 2*2*64), input_shape=(2, 2, 64)))
            model.add(Flatten())
            model.add(Dense(80, activation='relu', use_bias=True))
            model.add(Dropout(0.5))
            model.add(Dense(2))
            model.add(Activation('softmax'))
            myoptimizer = keras.optimizers.Adam(lr=0.001)
            model.compile(loss='mse', optimizer=myoptimizer, metrics=['accuracy'])

            methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, context, window_size, coverage_threshold)
            data_size = min(data_size, 2*len(methylated_train), 2*len(unmethylated_train))
            step = 20000
            slice = 0
            sample_set = methylated_train[slice:slice+step]+unmethylated_train[slice:slice+step]
            random.shuffle(sample_set)
            profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, window_size=window_size)
            X, Y = pg.data_preprocess(profiles, targets, include_annot=include_annot)
            b = np.zeros((Y.size, Y.max()+1))
            b[np.arange(Y.size), Y] = 1
            Y = b
            X = np.swapaxes(X, 1, 2)
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
            np.savetxt("GFG_mrcnn.csv", res, delimiter=", ", fmt='% s')


run_experiment_mrcnn([configs.Arabidopsis_config], ['CG', 'CHG', 'CHH'], 1600, (40, 40), 600000, coverage_threshold=10, include_annot=False)
