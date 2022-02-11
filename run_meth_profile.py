import configs
import meth_profile as mp
import meth_profile_cross_organism as mpc

config_list = config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
context_list = ['CG', 'CHG', 'CHH']
final_res = mp.experiments(config_list, context_list, dataset_size=50000, window_size=20, coverage_threshold=10)

#mpc.experiments(config_list, context_list, dataset_size=50000, window_size=20, coverage_threshold=10)
