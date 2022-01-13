import mrcnn
import configs as configs
import numpy as np
config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
contexts = ['CG', 'CHG', 'CHH']

cnfg = config_list[0]
res = []
for cnfg in config_list:
    for context in contexts:
        acc = mrcnn.run_experiment(cnfg, context)
        res.append([cnfg['organism_name'], context, acc])
        print('################################')
        print([cnfg['organism_name'], context, acc])
        np.savetxt("GFG_mrcnn_01.csv", res, delimiter =", ", fmt ='% s')
