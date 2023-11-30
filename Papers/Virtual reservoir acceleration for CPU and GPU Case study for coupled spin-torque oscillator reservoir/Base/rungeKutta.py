class RungeKutta:
    NAME = "rungeKutta"
    differential = 0
    stepSize = 0

    def run(self, state, time, stepSize, system):
        k1 = system.differential(state, time)
        k2 = system.differential(state+stepSize/2.0*k1, time+stepSize/2.0)
        k3 = system.differential(state+stepSize/2.0*k2, time+stepSize/2.0)
        k4 = system.differential(state+stepSize*k3, time+stepSize)

        return state + stepSize/6.0*(k1+2*k2+2*k3+k4)
