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
def get_profiles(methylations, sample_set, sequences_onehot, annot_seqs_onehot, window_size=3200):
    boundary_cytosines = 0
    profiles = np.zeros([len(sample_set), window_size, 4 + 2*len(annot_seqs_onehot)], dtype='short')
    targets = np.zeros(len(sample_set), dtype='short')
    total = len(sample_set)
    count = 0
    start = datetime.datetime.now()
    for index, position in enumerate(sample_set):
        row = methylations.iloc[position]
        center = row['position'] - 1
        chro = row['chr']
        targets[index] = round(float(row['meth']) / (row['meth'] + row['unmeth']))
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

def get_processed_data(cnfg):
    organism_name = cnfg['organism_name']
    methylations = data_reader.read_methylations(cnfg['methylation_address'])
    sequences = data_reader.readfasta(cnfg['seq_address'])
    annot_df = data_reader.read_annot(cnfg['annot_address'])
    if organism_name == configs.Cowpea_config['organism_name']:
        sequences = compatibility.cowpea_sequence_dic_key_compatibility(sequences)
        annot_df = compatibility.cowpea_annotation_compatibility(annot_df)
        methylations = compatibility.cowpea_methylation_compatibility(methylations)
    annot_seq_df_list = []
    annot_tag = ''
    for at in cnfg['annot_types']:
        annot_subset = preprocess.subset_annot(annot_df, at)
        annot_str = preprocess.make_annotseq_dic(organism_name, at, annot_subset, sequences, from_file=True)
        annot_seq_df_list.append(annot_str)
        annot_tag += at
    sequences_df = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=True)
    return sequences_df, methylations, annot_seq_df_list

def run_experiments(config_list, context_list, window_size, data_size, coverage_threshold=10):
    res = []
    logs = []
    logs.append(['organism_name', 'context', 'data_size', 'window_size', 'slice', 'me_sz', 'ume_sz', 'test_sample_size', 'sample_set', 'profiles', 'x_train', 'x_test', 'x_val', 'dtype'])
    for cnfg in config_list:
        organism_name = cnfg['organism_name']
        sequences_onehot, methylations, annot_seqs_onehot = get_processed_data(cnfg)
        for context in context_list:
            #two shuffle list of the positions of the methylated and unmethylated cytosines in methylations is generated.
            methylated, unmethylated = preprocess.methylations_subseter(methylations, context, window_size, coverage_threshold)
            me_sz = len(methylated)
            ume_sz = len(unmethylated)
            # if it is too large take 100,000. if it is small take the 10 percent of the methylated and unmethylated
            test_sample_size = int(min(50000, len(methylated)/10, len(unmethylated)/10))
            test_sample_set = methylated[:test_sample_size]+unmethylated[:test_sample_size]
            methylated = methylated[test_sample_size:]
            unmethylated = unmethylated[test_sample_size:]
            test_profiles, test_targets = get_profiles(methylations, test_sample_set, sequences_onehot, annot_seqs_onehot, window_size=3200)
            x_test, y_test = data_preprocess(test_profiles, test_targets, include_annot=True)
            PROFILE_ROWS = x_test.shape[1]
            PROFILE_COLS = x_test.shape[2]
            np.save('./temporary_files/x_test.npy', x_test)
            np.save('./temporary_files/y_test.npy', y_test)
            del test_profiles, test_targets, x_test, y_test
            data_size = min(data_size, 2*len(methylated), 2*len(unmethylated))
            block_size = (80, 40)
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
            step = 50000
            for slice in range(0, int(data_size/2), step):
                if step+slice > data_size:
                    break
                sample_set = methylated[slice:slice+step]+unmethylated[slice:slice+step]
                random.shuffle(sample_set)
                profiles, targets = get_profiles(methylations, sample_set, sequences_onehot, annot_seqs_onehot, window_size=3200)
                X, Y = data_preprocess(profiles, targets, include_annot=True)
                x_train, x_val, y_train, y_val = split_data(X, Y, pcnt=0.1)
                with tf.device('/device:GPU:0'):
                    model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
                x_test = np.load('./temporary_files/x_test.npy')
                y_test = np.load('./temporary_files/y_test.npy')
                y_pred = model.predict(x_test)
                logs.append([organism_name, context, data_size, window_size, slice, me_sz, ume_sz,
                             test_sample_size, len(sample_set), len(profiles), len(x_train), len(x_test), len(x_val), x_train.dtype])
                np.savetxt("logs.csv", logs, delimiter=", ", fmt='% s')
                step_res = [organism_name, context, 'seq-annot', window_size, slice, accuracy_score(y_test, y_pred.round()),
                            f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
                del x_test, y_test
                print(step_res)
                res.append(step_res)
                np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')

def data_preprocess(X, Y, include_annot=False):
    if not include_annot:
        X = np.delete(X, range(4, X.shape[2]), 2)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    return X, Y

def split_data(X, Y, pcnt=0.1):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=pcnt, random_state=None)
    return x_train, x_test, y_train, y_test

