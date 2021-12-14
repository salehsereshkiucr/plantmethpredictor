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


res = pg.run_experiments([configs.Cowpea_config], context_list, window_sizes, block_sizes, [0, 500000], coverage_threshold=10, include_annot=False)
np.savetxt("GFG1.csv", res, delimiter=", ", fmt='% s')
res = pg.run_experiments([configs.Cucumber_config], context_list, window_sizes, block_sizes, [0, 500000], coverage_threshold=10, include_annot=False)
np.savetxt("GFG2.csv", res, delimiter=", ", fmt='% s')
res = pg.run_experiments([configs.Tomato_config], context_list, window_sizes, block_sizes, [0, 500000], coverage_threshold=10, include_annot=False)
np.savetxt("GFG3.csv", res, delimiter=", ", fmt='% s')

