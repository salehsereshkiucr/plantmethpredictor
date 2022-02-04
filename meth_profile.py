import random
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
import profile_generator as pg
import configs as configs

def profiler(methylations, context, datasize, window_size=20):
    half_w = int(window_size/2)
    methylations = methylations.sort_values(["chr", "position"], ascending=(True, True))
    chrs_counts = methylations['chr'].value_counts()
    last_chr_pos = {}
    chrnums = list(chrs_counts.index)
    sum = 0
    for i in range(len(chrnums)):
        last_chr_pos[i] = sum+chrs_counts[i]-1
        sum += chrs_counts[i]
    # last_chr_pos ==> {0: 5524, 1: 1042784, 2: 1713034, 3: 2550983, 4: 3205486, 5: 4145381, 6: 4153872}
    # methylations.iloc[2550983] => chr 3.0 position    23459763.0
    # methylations.iloc[2550984] => chr 4.0 position    1007
    methylations.insert(0, 'idx', range(0, len(methylations)))
    sub_methylations = methylations[methylations['context'] == context]
    idxs = sub_methylations['idx']
    mlevels = methylations['mlevel']
    mlevels = np.asarray(mlevels)
    X = np.zeros((datasize, window_size))
    Y = np.zeros(datasize)
    avlbls = np.asarray(idxs)
    for lcp in list(last_chr_pos.values()):
        if lcp > 0 and lcp < len(mlevels) - window_size:
            avlbls = np.setdiff1d(avlbls, range(lcp-half_w, lcp+half_w))
    smple = random.sample(list(avlbls), datasize)
    for index, p in enumerate(smple):
        X[index] = np.concatenate((mlevels[p-half_w: p] , mlevels[p+1: p+half_w+1]), axis=0)
        Y[index] = 0 if mlevels[p] < 0.5 else 1
    X = X.reshape(list(X.shape) + [1])
    return X, Y

def run_experiment(X, Y, window_size=20, test_percent=0.2, test_val_percent = 0.5):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=None)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_val_percent, random_state=None)
    model = Sequential()
    model.add(Dense(window_size, activation='relu', input_shape=((window_size,1))))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0, validation_data=(x_val, y_val))
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred.round())

def experiments(config_list, context_list, dataset_size=50000, window_size=20, coverage_threshold=10):
    res = []
    for cnfg in config_list:
        methylations = pg.get_methylations(cnfg, '', coverage_threshold)
        for context in context_list:
            X, Y = profiler(methylations, context, dataset_size, window_size=window_size)
            acc = run_experiment(X, Y, window_size=20, test_percent=0.2, test_val_percent=0.5)
            res.append([cnfg['organism_name'], context, acc])
            np.savetxt("meth_profiles.csv", res, delimiter =", ", fmt ='% s')

config_list = config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
context_list = ['CG', 'CHG', 'CHH']
final_res = experiments(config_list, context_list, dataset_size=50000, window_size=20, coverage_threshold=10)
