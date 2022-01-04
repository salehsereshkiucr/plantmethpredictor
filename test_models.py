import configs as configs
from tensorflow import keras
import profile_generator as pg
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

context_list = [
    'CG',
    'CHG',
    'CHH']
cnfgs = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
include_annot = False
window_size = 3200
res = []
for cnfg in cnfgs:
    organism_name = cnfg['organism_name']
    for context in context_list:
        model = keras.models.load_model('./models/' + organism_name+context + '499983200.mdl')
        for test_config in cnfgs:
            test_organism = test_config['organism_name']
            if organism_name == test_organism:
                continue
            sequences_onehot, methylations, annot_seqs_onehot, num_to_chr_dic = pg.get_processed_data(test_config, context, coverage_threshold=10)
            x_test, y_test = pg.test_sampler(methylations, sequences_onehot, annot_seqs_onehot, window_size, num_to_chr_dic, include_annot=include_annot)
            tag = 'seq-only'
            if include_annot:
                tag = 'seq-annot'
            y_pred = model.predict(x_test)
            del sequences_onehot, methylations, annot_seqs_onehot, num_to_chr_dic
            step_res = [organism_name, test_organism, context, tag, window_size, 500000, len(x_test), accuracy_score(y_test, y_pred.round()),
                                f1_score(y_test, y_pred.round()), precision_score(y_test, y_pred.round()), recall_score(y_test, y_pred.round())]
            res.append(step_res)
            np.savetxt("GFG_cross_organism.csv", res, delimiter =", ", fmt ='% s')

