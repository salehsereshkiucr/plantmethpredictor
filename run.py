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
data_size = 600000
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
#res = pg.run_experiments(config_list, context_list, 3200, (80, 40), data_size, coverage_threshold=10)

res = pg.run_experiments(config_list, context_list, 3200, (80, 40), data_size, [0, 40000, 80000, 120000, 200000, 400000, 600000], coverage_threshold=10, include_annot=False)
