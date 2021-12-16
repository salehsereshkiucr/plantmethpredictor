import compatibility as compatibility
import configs as configs
import numpy as np
import data_reader as data_reader
import preprocess as preprocess
import random
import os
import datetime
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
from os import path

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

cnfgs = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Tomato_config, configs.Cucumber_config]
contexts = [
    'CG',
    'CHG',
    'CHH']
output_root = '/home/ssere004/SalDMR/predictordataprovider/output_complete/'

#This method gets for an organism and context, mekes
def get_profiles(methylations, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=3200):
    boundary_cytosines = 0
    profiles = np.zeros([len(sample_set), window_size, 4 + 2*len(annot_seqs_onehot)], dtype='short')
    targets = np.zeros(len(sample_set), dtype='short')
    total = len(sample_set)
    count = 0
    start = datetime.datetime.now()
    for index, position in enumerate(sample_set):
        row = methylations.iloc[position]
        center = int(row['position'] - 1)
        chro = num_to_chr_dic[row['chr']]
        targets[index] = round(float(row['mlevel']))
        try:
            profiles[index] = get_window_seqgene_df(sequences_onehot, annot_seqs_onehot, chro, center, window_size)
        except:
            boundary_cytosines += 1
        if count % int(total/10) == 0:
            now = datetime.datetime.now()
            seconds = (now - start).seconds
            print(str(int(count * 100/total)) + '%' + ' in ' + str(seconds) +' seconds')
        count += 1
    print(str(boundary_cytosines) + ' boundary cytosines are ignored')
    return profiles, targets



def get_window_seqgene_df(sequences_df, annot_seq_df_list, chro, center, window_size):
    profile_df = sequences_df[chro][center - int(window_size/2): center + int(window_size/2)]
    for i in range(len(annot_seq_df_list)):
        profile_df = np.concatenate([profile_df, annot_seq_df_list[i][chro][center - int(window_size/2): center + int(window_size/2)]], axis=1)
    return profile_df

def get_processed_data(cnfg, context, coverage_threshold=10, methylations_from_file=False, annotseqs_from_file=True, sequneces_df_from_file=True):
    organism_name = cnfg['organism_name']
    methylations = data_reader.read_methylations(cnfg['methylation_address'], context, coverage_threshold=coverage_threshold)
    sequences = data_reader.readfasta(cnfg['seq_address']) #Can get shrinked the size.
    annot_df = data_reader.read_annot(cnfg['annot_address'])
    if organism_name == configs.Cowpea_config['organism_name']:
        sequences = compatibility.cowpea_sequence_dic_key_compatibility(sequences)
        annot_df = compatibility.cowpea_annotation_compatibility(annot_df)
        methylations = compatibility.cowpea_methylation_compatibility(methylations)
    methylations, num_to_chr_dic = preprocess.shrink_methylation(methylations)
    methylations_train, methylations_test = preprocess.seperate_methylations(organism_name, methylations, from_file=methylations_from_file)
    annot_seq_df_list = []
    annot_tag = ''
    for at in cnfg['annot_types']:
        annot_subset = preprocess.subset_annot(annot_df, at)
        annot_str = preprocess.make_annotseq_dic(organism_name, at, annot_subset, sequences, from_file=annotseqs_from_file)
        annot_seq_df_list.append(annot_str)
        annot_tag += at
    sequences_df = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=sequneces_df_from_file)
    return sequences_df, methylations_train, methylations_test, annot_seq_df_list, num_to_chr_dic


def test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic, include_annot=False):
    methylated, unmethylated = preprocess.methylations_subseter(methylations_test, window_size)
    test_sample_size = int(min(50000, 2*len(methylated), 2*len(unmethylated)))
    test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
    random.shuffle(test_sample_set)
    test_profiles, test_targets = get_profiles(methylations_test, test_sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_size)
    x_test, y_test = data_preprocess(test_profiles, test_targets, include_annot=include_annot)
    return x_test, y_test


def run_experiments(config_list, context_list, window_sizes, block_sizes, steps, coverage_threshold=10, include_annot=True, memory_chunk_size=10000):
    res = []
    for cnfg in config_list:
        organism_name = cnfg['organism_name']
        for context in context_list:
            sequences_onehot, methylations_train, methylations_test, annot_seqs_onehot, num_to_chr_dic = get_processed_data(cnfg, context, coverage_threshold=coverage_threshold)
            for w in range(len(window_sizes)):
                PROFILE_ROWS = window_sizes[w]
                PROFILE_COLS = 4
                if include_annot:
                    PROFILE_COLS = 4 + 2*len(annot_seqs_onehot)
                model = Sequential()
                model.add(Conv2D(16, kernel_size=(1, PROFILE_COLS), activation='relu', input_shape=(PROFILE_ROWS, PROFILE_COLS, 1)))
                model.add(Reshape((block_sizes[w][0], block_sizes[w][1], 16), input_shape=(PROFILE_ROWS, 1, 16)))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))
                print('model processed')
                opt = tf.keras.optimizers.SGD(lr=0.01)
                model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
                methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, window_sizes[w])
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
                        step = ds_size * 2
                        last = True
                    slice = int(steps[s]/2)
                    for chunk in range(slice, slice+int(step/2), memory_chunk_size):
                        sample_set = methylated_train[chunk:chunk+memory_chunk_size]+unmethylated_train[chunk:chunk+memory_chunk_size]
                        random.shuffle(sample_set)
                        profiles, targets = get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=window_sizes[w])
                        X, Y = data_preprocess(profiles, targets, include_annot=include_annot)
                        x_train, x_val, y_train, y_val = split_data(X, Y, pcnt=0.1)
                        x_train_sz += len(x_train)
                        with tf.device('/device:GPU:0'):
                            model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))

                    x_test, y_test = test_sampler(methylations_test, sequences_onehot, annot_seqs_onehot, window_sizes[w], num_to_chr_dic, include_annot=include_annot)

                    y_pred = model.predict(x_test)
                    tag = 'seq-only'
                    if include_annot:
                        tag = 'seq-annot'
                    step_res = [organism_name, context, tag, window_sizes[w], x_train_sz, len(x_train), len(x_test), accuracy_score(y_test, y_pred.round()),
                            f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
                    del x_test, y_test
                    print(step_res)
                    res.append(step_res)
                    np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
    return res

def data_preprocess(X, Y, include_annot=False):
    if not include_annot:
        X = np.delete(X, range(4, X.shape[2]), 2)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    return X, Y

def split_data(X, Y, pcnt=0.1):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=pcnt, random_state=None)
    return x_train, x_test, y_train, y_test

