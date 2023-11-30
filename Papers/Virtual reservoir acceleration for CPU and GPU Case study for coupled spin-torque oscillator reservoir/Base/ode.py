import rungeKutta as rk

def makeSolver(solverName):
    if solverName == "rungeKutta":
        return rk.RungeKutta()

class Ode:
    def run(self, times, states, timeStep, system, solver):
        for i in range(times.size - 1):
            states[i+1] = solver.run(states[i], times[i], timeStep, system)