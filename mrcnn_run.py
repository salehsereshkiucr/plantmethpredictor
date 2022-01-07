import mrcnn
import configs as configs
config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
contexts = ['CG', 'CHG', 'CHH']

cnfg = config_list[0]
for context in contexts:
    print(cnfg['organism_name'], context, mrcnn.run_experiment(cnfg, context))
