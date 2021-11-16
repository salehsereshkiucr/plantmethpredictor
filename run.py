import configs as configs
import profile_generator as pg
import numpy as np
config_list = [configs.Arabidopsis_config]
context_list = [
    'CG',
    'CHG',
    'CHH']
window_size = 3200
data_size = 100000
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
res = pg.run_experiments(config_list, context_list, 3200, (80, 40), data_size, coverage_threshold=10)
#for w in range(len(window_sizes)):
#    res = pg.run_experiments(config_list, context_list, window_sizes[w], block_sizes[w], data_size, coverage_threshold=10)
#    for r in res:
#       final_res.append(r)
#    np.savetxt("GFG_final_res.csv", final_res, delimiter =", ", fmt ='% s')
