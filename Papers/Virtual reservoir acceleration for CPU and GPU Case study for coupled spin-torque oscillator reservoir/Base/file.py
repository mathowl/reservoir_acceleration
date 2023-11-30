import math
import numpy as np

import common

def loadLlgConfigs(llgConfigFileName, separator):
    tempLlgConfigs = np.genfromtxt(llgConfigFileName, dtype = "unicode")
    llgConfigs = {"magnetization": float(tempLlgConfigs[0])
                  , "demagCoef": np.array([float(tempLlgConfigs[1]), float(tempLlgConfigs[2]), float(tempLlgConfigs[3])])
                  , "magneticField": np.array([float(tempLlgConfigs[4]), float(tempLlgConfigs[5]), float(tempLlgConfigs[6])])
                  , "appliedField": float(tempLlgConfigs[7])
                  , "theta_H": common.arcToRad(float(tempLlgConfigs[8]))
                  , "phi_H": common.arcToRad(float(tempLlgConfigs[9]))
                  , "dip": float(tempLlgConfigs[10]) 
                  , "current": float(tempLlgConfigs[11]) 
                  , "spinPolarization": float(tempLlgConfigs[12])
                  , "torqueAsymmetry": float(tempLlgConfigs[13])
                  , "beta": float(tempLlgConfigs[14])
                  , "theta_p": common.arcToRad(float(tempLlgConfigs[15]))
                  , "phi_p": common.arcToRad(float(tempLlgConfigs[16]))
                  , "thickness": float(tempLlgConfigs[17])
                  , "radius": np.array([float(tempLlgConfigs[18]), float(tempLlgConfigs[19])])
                  , "gyro": float(tempLlgConfigs[20])
                  , "alpha": float(tempLlgConfigs[21])
                  , "temperature": float(tempLlgConfigs[22])
                  , "theta": common.arcToRad(float(tempLlgConfigs[23]))
                  , "phi": common.arcToRad(float(tempLlgConfigs[24]))
                }
    return llgConfigs

def loadOdeConfigs(odeConfigFileName, separator):
    tempOdeConfigs = np.genfromtxt(odeConfigFileName, dtype = "unicode")
    llgConfigs = {"solverName": tempOdeConfigs[0]
                  ,"totalTime": float(tempOdeConfigs[1])
                  , "stepSize": float(tempOdeConfigs[2])
                  , "samplingInterval": float(tempOdeConfigs[3])
                }
    return llgConfigs

def loadInputLlgConfigs(configFileName, separator):
    tempConfigs = np.genfromtxt(configFileName, dtype = "unicode")
    configs = {"inputCount": int(tempConfigs[0])
              , "inputScaleFactors": tempConfigs[1].split(',')
              , "theta_input": common.arcToRad(float(tempConfigs[2]))
              , "phi_input": common.arcToRad(float(tempConfigs[3]))
              , "seed": int(tempConfigs[4])
              }

    return configs
    
def loadCoupledLlgConfigs(configFileName, separator):
    tempConfigs = np.genfromtxt(configFileName, dtype = "unicode")
    configs = {"stoCount": int(tempConfigs[0])
              , "spectorRadius": float(tempConfigs[1])
              , "theta_coupling": common.arcToRad(float(tempConfigs[2]))
              , "phi_coupling": common.arcToRad(float(tempConfigs[3]))
              , "seed": int(tempConfigs[4])
              }
    return configs


#コマンドラインコンフィグに値が存在すればコンフィグに代入
def loadCommandLineConfigs(configs, commandLineConfigs):
    configNames = list(configs)
    for configName in configNames:
        if configName in commandLineConfigs:
            configs[configName] = commandLineConfigs[configName]