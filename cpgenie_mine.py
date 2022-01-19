from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.constraints import max_norm
import numpy as np


import preprocess as preprocess
import profile_generator as pg
import random
from datetime import datetime
import tensorflow as tf

def test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic, include_annot=False):
    methylated, unmethylated = preprocess.methylations_subseter(methylations_test, window_size)
    test_sample_size = int(min(50000, 2*len(methylated), 2*len(unmethylated)))
    test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
    random.shuffle(test_sample_set)
    test_profiles, test_targets = pg.get_profiles(methylations_test, test_sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
    x_test, y_test = preprocess.cpgenie_preprocess(test_profiles, test_targets)
    return x_test, y_test

def run_experiments(config_list, context_list, steps, coverage_threshold=10, include_annot=True, memory_chunk_size=10000):
    res = []
    for cnfg in config_list:
        organism_name = cnfg['organism_name']
        for context in context_list:
            sequences_onehot, methylations, annot_seqs_onehot, num_to_chr_dic = pg.get_processed_data(cnfg, context, coverage_threshold=coverage_threshold)
            methylations_train, methylations_test = preprocess.seperate_methylations(organism_name, methylations, from_file=False)
            if not include_annot:
                del annot_seqs_onehot
                annot_seqs_onehot = []
            PROFILE_ROWS = 1000
            PROFILE_COLS = 4

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

            methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, 1000)
            ds_size = min(len(methylated_train), len(unmethylated_train))
            x_train_sz = 0
            last = False
            for s in range(len(steps) - 1):
                if last:
                    break
                step = steps[s+1] - steps[s]
                print('##################################', step)
                print('#################################', ds_size)
                if ds_size * 2 < steps[s+1]: #temporary, does not work for dataset size checking.
                    step = (ds_size * 2) - 2
                    last = True
                slice = int(steps[s]/2)
                for chunk in range(slice, slice+int(step/2), memory_chunk_size):
                    if chunk+memory_chunk_size > slice+int(step/2):
                        sample_set = methylated_train[chunk:slice+int(step/2)]+unmethylated_train[chunk:slice+int(step/2)]
                    else:
                        sample_set = methylated_train[chunk:chunk+memory_chunk_size]+unmethylated_train[chunk:chunk+memory_chunk_size]
                    random.shuffle(sample_set)
                    profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=1000)
                    X, Y = preprocess.cpgenie_preprocess(profiles, targets)
                    x_train, x_val, y_train, y_val = pg.split_data(X, Y, pcnt=0.1)
                    x_train_sz += len(x_train)
                    with tf.device('/device:GPU:0'):
                        print('model fitting started for ' + organism_name + ' ' + context)
                        print(datetime.now())
                        model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=0, validation_data=(x_val, y_val))
                        print('model fitting ended for ' + str(len(x_train)) + ' data')
                        print(datetime.now())
                        del x_train, y_train
                model_tag = str(organism_name) + str(context) + str(x_train_sz) + str(1000) + '.mdl'
                model.save('./models/' + model_tag)

                x_test, y_test = pg.test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, 1000, num_to_chr_dic, include_annot=include_annot)
                y_pred = model.predict(x_test)
                tag = 'seq-only'
                if include_annot:
                    tag = 'seq-annot'
                step_res = [organism_name, context, tag, 1000, x_train_sz, len(x_test), accuracy_score(y_test, y_pred.round()),
                        f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
                del x_test, y_test
                print(step_res)
                print(datetime.now())
                res.append(step_res)
                np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
    return res
