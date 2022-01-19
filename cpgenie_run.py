import configs as configs
import cpgenie_mine as cpgm
context_list = [
    'CG',
    'CHG',
    'CHH']
config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]


cpgm.run_experiments(config_list, context_list, [0, 10000], coverage_threshold=10, include_annot=True, memory_chunk_size=10000)
