import mrcnn
import configs as configs
import numpy as np
config_list = [configs.Arabidopsis_config, configs.Cowpea_config, configs.Rice_config, configs.Cucumber_config, configs.Tomato_config]
contexts = ['CG', 'CHG', 'CHH']

cnfg = config_list[0]
res = []
for context in contexts:
    res.append([cnfg['organism_name'], context, mrcnn.run_experiment(cnfg, context)])
    print(cnfg['organism_name'], context, mrcnn.run_experiment(cnfg, context))
    np.savetxt("GFG_final_res.csv", res, delimiter =", ", fmt ='% s')
