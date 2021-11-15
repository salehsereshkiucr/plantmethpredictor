import configs as configs
import profile_generator as pg
config_list = [configs.Arabidopsis_config]
context_list = [
    #'CG',
    'CHG',
    'CHH']
window_size = 3200
data_size = 500000
pg.run_experiments(config_list, context_list, window_size, data_size, coverage_threshold=10)
