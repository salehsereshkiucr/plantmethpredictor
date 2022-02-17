import configs as configs
import cpgenie_mine as cpgm
context_list = [
    'CHG',
    'CHH']

config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
config_list = [configs.Rice_config]


cpgm.run_experiments(config_list, ['CHH'], [0, 500000], coverage_threshold=10, include_annot=False, memory_chunk_size=10000)
