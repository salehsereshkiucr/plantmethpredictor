import configs as configs
import profile_generator as pg
config_list = [configs.Arabidopsis_config]
context_list = [
    'CG',
    'CHG',
    'CHH']
window_size = 3200
data_size = 500000
window_sizes = [
    100,
    200,
    400,
    800,
    1600,
    3200
]
block_sizes = [(10, 10), (20, 10), (20, 20), (40, 20), (40, 40), (80, 40)]
for w in range(len(window_sizes)):
    pg.run_experiments(config_list, context_list, window_sizes[w], block_sizes[w], data_size, coverage_threshold=10)
