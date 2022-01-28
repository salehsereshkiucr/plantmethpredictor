from sklearn.ensemble import RandomForestClassifier
import numpy as np
import configs as configs
from sklearn import metrics
import preprocess as preprocess
import profile_generator as pg
import random as random

config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
context_list = [
    'CG',
    'CHG',
    'CHH']
coverage_threshold = 10
ds = 500000
res = []
for cnfg in config_list:
    for context in context_list:
        organism_name = cnfg['organism_name']
        sequences_onehot, methylations, annot_seqs_onehot, num_to_chr_dic = pg.get_processed_data(cnfg, context, coverage_threshold=coverage_threshold)
        methylations_train, methylations_test = preprocess.seperate_methylations(organism_name, methylations, from_file=False)
        methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations_train, 3200)
        available_ds = min(len(methylated_train), len(unmethylated_train)) * 2
        final_ds = min(available_ds, ds)
        sample_set = methylated_train[0:int(final_ds/2)]+unmethylated_train[0:int(final_ds/2)]
        random.shuffle(sample_set)
        profiles, targets = pg.get_profiles(methylations_train, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=3200)
        X, Y = pg.data_preprocess(profiles, targets, include_annot=False)
        x_train, x_test, y_train, y_test = pg.split_data(X, Y, pcnt=0.1)
        clf = RandomForestClassifier(random_state=0, n_estimators=50, warm_start=True, n_jobs=-1)
        for chunk in range(0, len(x_train), 10000):
            if chunk+10000 < len(x_train):
                xx = x_train[chunk: chunk+10000]
                nsamples, nx, ny, nz = xx.shape
                xx = xx.reshape((nsamples, nx*ny))
                yy = y_train[chunk: chunk+10000]
                clf.fit(xx, yy)
                clf.n_estimators += 50
        nsamples, nx, ny, nz = x_test.shape
        x_test = x_test.reshape((nsamples, nx*ny))
        y_pred=clf.predict(x_test)
        step_res = ["organism_name", context, final_ds, metrics.accuracy_score(y_test, y_pred)]
        res.append(step_res)
        print(step_res)
        np.savetxt("GFG.csv", res, delimiter =", ", fmt ='% s')
