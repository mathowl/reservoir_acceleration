import os
import time
import logging
import shelve
import numpy as np

import matplotlib.pyplot as plt

import ode
import rungeKutta as rk

import common
import file
import plot
#import Llg
import coupledLlg

def runCoupledLlg(commandLineConfigs):
#logging
    logging.basicConfig(filename='status.log', level=logging.INFO, format=common.LOG_FORMAT)
    initialTime = time.time()

#load configs in files
    odeConfigs = file.loadOdeConfigs(common.ODE_CONFIG_FILE_NAME, " ")
    llgConfigs = file.loadLlgConfigs(common.LLG_CONFIG_FILE_NAME, " ")
    coupledLlgConfigs = file.loadCoupledLlgConfigs(common.COUPLED_LLG_CONFIG_FILE_NAME, " ")
    inputLlgConfigs = file.loadInputLlgConfigs(common.INPUT_LLG_CONFIG_FILE_NAME, " ")

#load configs in commandLine
    file.loadCommandLineConfigs(odeConfigs, commandLineConfigs)
    file.loadCommandLineConfigs(llgConfigs, commandLineConfigs)
    file.loadCommandLineConfigs(coupledLlgConfigs, commandLineConfigs)
    file.loadCommandLineConfigs(inputLlgConfigs, commandLineConfigs)

#initialization
    system = coupledLlg.CoupledLlg(llgConfigs, inputLlgConfigs, coupledLlgConfigs)
    system.normalization(odeConfigs["stepSize"])

    stepCount = int(odeConfigs["totalTime"]/odeConfigs["stepSize"])
    samplingCount = int(odeConfigs["totalTime"]/odeConfigs["samplingInterval"])
    samplingFreq = int(odeConfigs["samplingInterval"]/odeConfigs["stepSize"])
    
    #TJ: different initial state
    initialState = np.tile(common.sphericalCoordinateToCartesianCoordinate(np.array([1, llgConfigs["theta"], llgConfigs["phi"]])), (system.stoCount, 1)) 

    states = np.zeros([stepCount, system.stoCount, 3])
    
    states[0] = initialState
    times = np.linspace(0, odeConfigs["totalTime"], stepCount, endpoint=False)
    solver = ode.makeSolver(odeConfigs["solverName"]) 

    logging.info("Start cluculation: " + str(time.time()) + "System: " + system.NAME + ", Solver: " + solver.NAME)

#solve ode
    myOde = ode.Ode()

    myOde.run(times, states, system.dt, system, solver)

    norms = np.linalg.norm(states, axis = 1)
    t_dur = str(time.time() - initialTime)
    logging.info("total time: " + str(time.time() - initialTime))
    print(t_dur)
    with shelve.open('perform_base.shelve', 'c') as shelf:
        shelf[str(time.time())] = [int(system.stoCount),t_dur]

    with shelve.open('sol_base.shelve', 'c') as shelf:
        shelf[str(int(system.stoCount))] = states[:,0,:]


    states_sampling = states[::samplingFreq]

    times_sampling = times[::samplingFreq]
    norms_sampling = np.linalg.norm(states_sampling, axis = 2)

#outputFile
    os.makedirs(common.RESULT_DIR_NAME, exist_ok=True)
    fileName = common.RESULT_DIR_NAME + "/trajectory"
    for key, value in commandLineConfigs.items():
        fileName = fileName + "_" + str(key) + "=" + str(value)
    np.savetxt(fileName + ".txt", np.concatenate([times_sampling.reshape([-1,1]), states_sampling.reshape([states_sampling.shape[0],-1]), norms_sampling.reshape([states_sampling.shape[0],-1])], axis=1))

#plot file
    os.makedirs(common.FIGURE_DIR_NAME, exist_ok=True)

    fileName = common.FIGURE_DIR_NAME +"/timeSeries_"
    for key, value in commandLineConfigs.items():
        fileName = fileName + "_" + str(key) + "=" + str(value)
    plot.plot_multiTimeseries(times_sampling, states_sampling, fileName)

    fileName = common.FIGURE_DIR_NAME +"/trajectory_"
    for key, value in commandLineConfigs.items():
        fileName = fileName + "_" + str(key) + "=" + str(value)
    plot.plot_multiTrajectory(states_sampling, fileName)

    fileName = common.FIGURE_DIR_NAME +"/norm_"
    for key, value in commandLineConfigs.items():
        fileName = fileName + "_" + str(key) + "=" + str(value)
    plot.plot_multiQuantity(times_sampling, norms_sampling, fileName)
    

   




