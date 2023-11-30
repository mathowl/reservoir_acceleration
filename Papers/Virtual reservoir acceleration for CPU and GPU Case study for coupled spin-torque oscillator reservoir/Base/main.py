import os
import time

import common
import runCoupledLlg

if __name__ == "__main__":
    os.makedirs(common.RESULT_DIR_NAME, exist_ok=True)
    os.makedirs(common.FIGURE_DIR_NAME, exist_ok=True)
    stoCounts = [1,10,100,1000,2500,5000,10_000]
    commandLineConfigs = [0]*len(stoCounts)

    for i in range(len(stoCounts)):
        commandLineConfigs[i] = {}
        commandLineConfigs[i]["stoCount"] = stoCounts[i]

    t_dict = dict()
    
    for commandLineConfig in commandLineConfigs:
        for i in range(0,4):
            runCoupledLlg.runCoupledLlg(commandLineConfig)    

