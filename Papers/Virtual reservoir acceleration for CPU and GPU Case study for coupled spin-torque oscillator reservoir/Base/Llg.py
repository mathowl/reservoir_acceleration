import math
from sre_parse import State

import numpy as np

import common

class Llg:
#system parameter
    NAME = "llg"
    DIMENSION = 3


    
    def __init__(self, llgConfigs):
        self.magnetization = llgConfigs["magnetization"]
        self.anisotropyField = llgConfigs["magneticField"] - 4.0 * self.magnetization * math.pi * llgConfigs["demagCoef"]
        self.appliedField = llgConfigs["appliedField"]
        self.theta = llgConfigs["theta_H"]
        self.phi = llgConfigs["phi_H"]
        self.dip = llgConfigs["dip"]
        self.current = llgConfigs["current"]
        self.spinPolarization = llgConfigs["spinPolarization"]
        self.torqueAsymmetry = llgConfigs["torqueAsymmetry"]
        self.beta = llgConfigs["beta"]
        self.pinned = common.sphericalCoordinateToCartesianCoordinate(np.array([1, llgConfigs["theta_p"], llgConfigs["phi_p"]]))
        self.volume = llgConfigs["thickness"] * math.pi * llgConfigs["radius"][0] * llgConfigs["radius"][1]
        self.gyro = llgConfigs["gyro"]
        self.alpha = llgConfigs["alpha"]
        self.temperature = llgConfigs["temperature"]

    def normalization(self, stepSize):
        fieldNormalizationFactor = np.amax(np.abs(self.anisotropyField))
        self.dt = (self.gyro * fieldNormalizationFactor * stepSize)
        #self.dtNano = (self.totalTime * 1.0e9) / common.MESH_TIME

        self.applied = common.sphericalCoordinateToCartesianCoordinate(np.array([1, self.theta, self.phi])) * self.appliedField / fieldNormalizationFactor
        self.dipoleField = self.dip / fieldNormalizationFactor * np.array([-1, -1, 2])
        self.h_sst = common.H_BAR * self.spinPolarization * self.current  / (2.0 * common.CHARGE * self.magnetization * self.volume * fieldNormalizationFactor)

        #self.randomField = math.sqrt(2.0 * self.alpha * common.K_B * self.temperature / (self.magnetization * self.volume * fieldNormalizationFactor * self.dt))
        self.anisotropy = self.anisotropyField / fieldNormalizationFactor

    def differential(self, state, time):
        # also different
        asymmetricFactor = 1.0 / (1.0 + self.torqueAsymmetry * np.dot(state, self.pinned))
        temp1 = self.dipoleField * self.pinned
        temp2 = self.anisotropy * state 

        #print(np.cross(self.pinned, state))
        #print((asymmetricFactor * self.h_sst) * (np.cross(self.pinned, state) + self.beta * self.pinned))
        
        # unclear to me where temp1 and self.beta * self.pinned
        b = self.applied + temp2 + (asymmetricFactor * self.h_sst) * (np.cross(self.pinned, state))
        #b = self.applied + temp1 + temp2 + (asymmetricFactor * self.h_sst) * (np.cross(self.pinned, state) + self.beta * self.pinned)
        c = np.cross(state, b)

        return (-1 / (1 + self.alpha*self.alpha)) * (c + self.alpha * np.cross(state, c))