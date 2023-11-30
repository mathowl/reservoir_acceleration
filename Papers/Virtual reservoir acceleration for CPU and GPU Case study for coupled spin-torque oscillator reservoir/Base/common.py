import math
import numpy as np

#probram configs
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s :%(message)s"

#physical constant value
H_BAR = 1.05457266e-27
CHARGE = 1.60217733e-19
K_B = 1.3806504e-16

#dir name
RESULT_DIR_NAME = "result"
FIGURE_DIR_NAME = "figure"

#file name
ODE_CONFIG_FILE_NAME = "data/odeConfig.txt"
LLG_CONFIG_FILE_NAME = "data/llgConfig.txt"
COUPLED_LLG_CONFIG_FILE_NAME = "data/coupledLlgConfig.txt"
INPUT_LLG_CONFIG_FILE_NAME = "data/inputLlgConfig.txt"

def sphericalCoordinateToCartesianCoordinate(sphiricalCoordinate):
    return sphiricalCoordinate[0]*np.array([math.sin(sphiricalCoordinate[1])*math.cos(sphiricalCoordinate[2])
                                          , math.sin(sphiricalCoordinate[1])*math.sin(sphiricalCoordinate[2])
                                          , math.cos(sphiricalCoordinate[1])])

def arcToRad(arc):
    return arc*math.pi/180.0

def vectorizeOuterProd (a, b):
    
    if a.ndim == 1:
        newA = np.tile(a, (b.shape[0], 1))
    else: newA = a

    ans = np.zeros(b.shape)
    ans[:,0] = newA[:,1] * b[:,2] - newA[:,2] * b[:,1]
    ans[:,1] = newA[:,2] * b[:,0] - newA[:,0] * b[:,2]
    ans[:,2] = newA[:,0] * b[:,1] - newA[:,1] * b[:,0]
    return ans

