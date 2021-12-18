import configs as configs
import profile_generator as pg
import preprocess as preprocess

import numpy as np
config_list = [configs.Rice_config]
context_list = [
    'CG',
    'CHG',
    'CHH']
window_size = 3200
window_sizes = [
    100,
    200,
    400,
    800,
    1600,
    3200
]
block_sizes = [(10, 10), (20, 10), (20, 20), (40, 20), (40, 40), (80, 40)]
final_res = []

steps = [0, 40000, 80000, 120000, 200000, 400000, 600000, 800000, 1000000]
train_config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
test_config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]

#for i in range(6):
#    res = pg.run_experiments([configs.Arabidopsis_config], context_list, [3200], [(80, 40)], steps, coverage_threshold=10, include_annot=False)
#    np.savetxt("GFG" + str(i) + ".csv", res, delimiter=", ", fmt='% s')


res = pg.run_experiments(train_config_list, context_list, [3200], [(80, 40)], [0, 500000], coverage_threshold=10, include_annot=False, cross_config=True, cnfg_test_list=test_config_list)
np.savetxt("GFG" + 'cross organism' + ".csv", res, delimiter=", ", fmt='% s')
