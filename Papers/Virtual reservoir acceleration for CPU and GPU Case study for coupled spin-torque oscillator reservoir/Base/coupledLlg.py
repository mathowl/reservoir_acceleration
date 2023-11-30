import math
#from sre_parse import State

import numpy as np
import time
import common

class CoupledLlg:
#system parameter
    NAME = "coupledLlg"

    
    def __init__(self, configs, inputLlgConfigs, coupledLlgConfigs):
        self.magnetization = configs["magnetization"]
        self.anisotropyField = configs["magneticField"] - 4.0 * self.magnetization * math.pi * configs["demagCoef"]
        self.appliedField = configs["appliedField"]
        self.dip = configs["dip"]
        self.current = configs["current"]
        self.spinPolarization = configs["spinPolarization"]
        self.torqueAsymmetry = configs["torqueAsymmetry"]
        self.beta = configs["beta"]
        self.pinned = common.sphericalCoordinateToCartesianCoordinate(np.array([1, configs["theta_p"], configs["phi_p"]]))
        self.volume = configs["thickness"] * math.pi * configs["radius"][0] * configs["radius"][1]
        self.gyro = configs["gyro"]
        self.alpha = configs["alpha"]
        self.temperature = configs["temperature"]

         


    #coupling configurations
        #self.inputWeight = 2*np.random.rand(self.stoCount, len(self.stoCount))-1
        self.stoCount = coupledLlgConfigs["stoCount"]
        np.random.seed(coupledLlgConfigs["seed"])
        self.seed = coupledLlgConfigs["seed"]
        self.dimension = self.stoCount*3

        self.appliedUnitVector = common.sphericalCoordinateToCartesianCoordinate(np.array([1, configs["theta_H"], configs["phi_H"]]))
        self.couplingAppliedUnitVector = common.sphericalCoordinateToCartesianCoordinate(np.array([1, coupledLlgConfigs["theta_coupling"], coupledLlgConfigs["phi_coupling"]]))
        
        if self.stoCount == 1:
            self.internalWeight = np.zeros([1,1])
        else:
            self.internalWeight = 2*np.random.rand(self.stoCount, self.stoCount)-1
        #no self coupling
            for i in range(self.stoCount):
                self.internalWeight[i, i] = 0
        #set spector radius
            eigenValues,__ = np.linalg.eig(self.internalWeight)
            self.internalWeight *= coupledLlgConfigs["spectorRadius"]/np.amax(np.abs(eigenValues))
            self.thetaCoupling = coupledLlgConfigs["theta_coupling"]
            self.phiCoupling = coupledLlgConfigs["phi_coupling"]

    #input configurations
        self.inputWeight = 2*np.random.rand(self.stoCount, inputLlgConfigs["inputCount"])-1
        self.inputAppliedUnitVector = np.tile(common.sphericalCoordinateToCartesianCoordinate(np.array([1, inputLlgConfigs["theta_input"], inputLlgConfigs["phi_input"]])), (self.stoCount, 1))

    #scale input weight
        for i in range(inputLlgConfigs["inputCount"]):
            self.inputWeight[:, i] = self.inputWeight[:, i] * coupledLlgConfigs["inputScaleFactors"][i]
            
    def normalization(self, stepSize):
        self.fieldNormalizationFactor = np.amax(np.abs(self.anisotropyField))
        self.dt = (self.gyro * self.fieldNormalizationFactor * stepSize)
        #self.dtNano = (self.totalTime * 1.0e9) / common.MESH_TIME

        #self.applied = common.sphericalCoordinateToCartesianCoordinate(np.array([1, self.theta, self.phi])) * self.appliedField / self.fieldNormalizationFactor
        self.pinnedMultipleddipoleField = self.dip / self.fieldNormalizationFactor * np.array([-1, -1, 2]) * self.pinned
        self.h_sst = common.H_BAR * self.spinPolarization * self.current  / (2.0 * common.CHARGE * self.magnetization * self.volume * self.fieldNormalizationFactor)

        #self.randomField = math.sqrt(2.0 * self.alpha * common.K_B * self.temperature / (self.magnetization * self.volume * fieldNormalizationFactor * self.dt))
        self.anisotropy = self.anisotropyField / self.fieldNormalizationFactor

#without input
    def differential(self, state, times):

        
        applied = (self.appliedField * self.appliedUnitVector).reshape(1, -1)
        coupledApplied =  (self.internalWeight @ state[:,0]).reshape(-1,1) @ self.couplingAppliedUnitVector.reshape(1, -1)
        applied = (applied + coupledApplied)/ self.fieldNormalizationFactor

        #asymmetricFactor = np.tile(1.0 / (1.0 + self.torqueAsymmetry * state @ self.pinned), (3,1)).T
        asymmetricFactor = 1.0 / (1.0 + self.torqueAsymmetry * state @ self.pinned).reshape(-1,1)
        temp = self.anisotropy * state

        #TJ:   self.pinnedMultipleddipoleField.reshape(1,-1) =0 , self.beta * self.pinned.reshape(1,-1) =0  
        
        b = applied + self.pinnedMultipleddipoleField.reshape(1,-1) + temp + (asymmetricFactor * self.h_sst) * (common.vectorizeOuterProd(self.pinned, state) + self.beta * self.pinned.reshape(1,-1))
        c = common.vectorizeOuterProd(state, b)
        return (-1 / (1 + self.alpha*self.alpha)) * (c + self.alpha * common.vectorizeOuterProd(state, c))

# with input
    #def differential(self, state, time, input):
    #    coupledApplied = self.internalWeight @ state[:,0].reshape([-1])
    #    inputApplied = self.inputWeight @ input
    #    self.applied = self.appliedUnitVector * self.appliedField + self.couplingUnitVector * coupledApplied + self.inputUnitVector * inputApplied/ self.fieldNormalizationFactor

    #    asymmetricFactor = np.tile(1.0 / (1.0 + self.torqueAsymmetry * state @ self.pinned), (3,1)).T
    #    temp = self.anisotropy * state

    #    b = self.applied + np.tile(self.pinnedMultipleddipoleField,(self.stoCount, 1)) + temp + (asymmetricFactor * self.h_sst) * (common.vectorizeOuterProd(self.pinned, state) + self.beta * np.tile(self.pinned, (self.stoCount, 1)))
    #    c = common.vectorizeOuterProd(state, b)
    #    return (-1 / (1 + self.alpha*self.alpha)) * (c + self.alpha * common.vectorizeOuterProd(state, c))