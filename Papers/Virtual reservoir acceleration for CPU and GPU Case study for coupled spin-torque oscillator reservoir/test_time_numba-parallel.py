import time
import numpy as np
import math
import shelve
import numba as nb


@nb.njit()
def Dfnumba(m,Wcp,P,pvec):   
    b=np.zeros((m.shape[0],3),dtype=m.dtype) 
    b[:,2] =  (P[7]  + P[2] * m[:,2])
    b[:,0] = P[3]*nb_matmul(Wcp, m[:,0]).T
    b-= np.multiply(calc_cros0( m, pvec),  (P[5])/(1+P[6] * calc_dot0(m,pvec)))
    c=calc_cros(m, b)
    return (-1 * c *P[1] - (P[0])*P[1]  * calc_cros(m , c))

@nb.njit(parallel=True)
def nb_matmul( A, b):
    res = np.empty(A.shape[0],dtype='float64')
    par = int(A.shape[1]/10)
    for i in nb.prange(par):
        r0,r1,r2,r3,r4,r5,r6,r7,r8,r9=0,0,0,0,0,0,0,0,0,0
        for j in nb.prange(0, A.shape[1]):
            r0+=A[i,j]*b[j]        
            r1+=A[i+par,j]*b[j]        
            r2+=A[i+2*par,j]*b[j]        
            r3+=A[i+3*par,j]*b[j]        
            r4+=A[i+4*par,j]*b[j]        
            r5+=A[i+5*par,j]*b[j]        
            r6+=A[i+6*par,j]*b[j]        
            r7+=A[i+7*par,j]*b[j]        
            r8+=A[i+8*par,j]*b[j]        
            r9+=A[i+9*par,j]*b[j]        
        res[i],res[i+par],res[i+2*par],res[i+3*par],res[i+4*par],res[i+5*par],res[i+6*par],res[i+7*par],res[i+8*par],res[i+9*par] = r0,r1,r2,r3,r4,r5,r6,r7,r8,r9
    return res


@nb.njit()
def calc_dot0(vec_1,vec_2):
    res=np.empty((vec_1.shape[0],1),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        res[i,0]=vec_1[i,0] * vec_2[0]+ vec_1[i,1] * vec_2[1] + vec_1[i,2] * vec_2[2]
    return res


@nb.njit()
def calc_cros0(vec_1,vec_2):
    res=np.empty((vec_1.shape[0],3),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        res[i,0]=vec_1[i,1] * vec_2[2] - vec_1[i,2] * vec_2[1]
        res[i,1]=vec_1[i,2] * vec_2[0] - vec_1[i,0] * vec_2[2]
        res[i,2]=vec_1[i,0] * vec_2[1] - vec_1[i,1] * vec_2[0]
   
    return res

@nb.njit()
def calc_cros(vec_1,vec_2):
    res=np.empty((vec_1.shape[0],3),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        res[i,0]=vec_1[i,1] * vec_2[i,2] - vec_1[i,2] * vec_2[i,1]
        res[i,1]=vec_1[i,2] * vec_2[i,0] - vec_1[i,0] * vec_2[i,2]
        res[i,2]=vec_1[i,0] * vec_2[i,1] - vec_1[i,1] * vec_2[i,0]
    return res



def maxeigPower(A,max_iter=1000,tol=1e-6):
    
    x = np.array(np.random.random(len(A))).T
    lam_prev = 0
    
    for i in range(max_iter):
        # Compute the updated approximation for the eigenvector
        x = A @ x / np.linalg.norm(A @ x)

        # Compute the updated approximation for the largest eigenvalue
        lam = (x.T @ A @ x) / (x.T @ x)

        # Check if the approximations have converged
        if np.abs(lam - lam_prev) < tol:
            break

        # Store the current approximation for the largest eigenvalue
        lam_prev = lam
    return lam


def genWcp(N):
    if N != 1:
        Wdum =np.random.random((N,N))*2-1
        Wdum = Wdum - np.diagflat(np.diagonal(Wdum))
        return Wdum/(np.abs(maxeigPower(Wdum)))
    else:
        return np.array([0])

def genWin(N,Nin):
        return np.random.random((N,Nin))*2-1

def sphericalCoordinateToCartesianCoordinate(sphiricalCoordinate):
    return sphiricalCoordinate[0]*np.array([math.sin(sphiricalCoordinate[1])*math.cos(sphiricalCoordinate[2])
                                          , math.sin(sphiricalCoordinate[1])*math.sin(sphiricalCoordinate[2])
                                          , math.cos(sphiricalCoordinate[1])])

@nb.njit()
def frk4(y,Wcp,h,P,pvec,total_steps):
    Y=np.empty((total_steps+1,3),dtype='float64')
    Y[0,:] = y[0,:]
    for i in nb.prange(total_steps):
        k1 = h * Dfnumba(y,Wcp,P,pvec)
        k2 = h * Dfnumba(y + 0.5 * k1,Wcp,P,pvec)
        k3 = h * Dfnumba( y + 0.5 * k2,Wcp,P,pvec)
        k4 = h * Dfnumba(y + k3,Wcp,P,pvec)
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        Y[i+1,:] = y[0,:]
    return Y





def Minitial(N):
    
    return np.array([list(sphericalCoordinateToCartesianCoordinate([1,2*math.pi/360,2*math.pi/360]))]*N)

def sphericalCoordinateToCartesianCoordinate(sphiricalCoordinate):
    return sphiricalCoordinate[0]*np.array([math.sin(sphiricalCoordinate[1])*math.cos(sphiricalCoordinate[2])
                                          , math.sin(sphiricalCoordinate[1])*math.sin(sphiricalCoordinate[2])
                                          , math.cos(sphiricalCoordinate[1])])









def main(N):
    P = np.array([0.005, \
        1/(1+0.005**2),\
        (18.616e3)   - 4*np.pi*1448.3,\
        1,\
        8500,\
        1.05457266e9*2.5*0.537/(2*1.60217733*720*np.pi*1448.3), \
        0.288,\
        200,\
        0,\
        416.12543922361147] ,dtype='float64')

    P1 =np.array([0.005, 1/(1+0.005**2),((18.616e3)   - 4*np.pi*1448.3)/P[9], \
                  1/P[9], 8500,1.05457266e9*2.5*0.537/(2*1.60217733*720*np.pi*1448.3*P[9]),\
                  0.288,200/P[9]],dtype='float64')



    pvec= np.array([1.000000e+00, 0.000000e+00, 6.123234e-17],dtype='float64')
    nsteps=5000
    h =  17640000.0 * 1e-11 * P[9]

    
    np.random.seed(0)

    Wcp0 =  2*np.random.rand(N,N)-1
    if N !=1:
        for i in range(N):
            Wcp0[i,i] = 0
        eigenValues,__ = np.linalg.eig(Wcp0)
        Wcp0 *= 1/np.amax(np.abs(eigenValues))
    else:
        Wcp0 = np.array([[0]])


    #compile
    M0 = Minitial(N)
    calc_dot0(M0,pvec)
    calc_cros(M0,M0)
    calc_cros0(M0,pvec)
    nb_matmul(Wcp0, M0[:,0])
    Dfnumba(M0,Wcp0,P1,pvec)
    frk4(M0,Wcp0,0.01,P1,pvec,1)


    #run code
    t=time.time()
    Y=frk4(M0,Wcp0,h,P1,pvec,nsteps-1)
    t_dur = float(time.time() - t)
    print(t_dur)
    with shelve.open('shelve/perform_numba-parallel.shelve', 'c') as shelf:
        shelf[str(time.time())] = [N,t_dur]
    with shelve.open('shelve/sol_numba-parallel.shelve', 'c') as shelf:
        shelf[str(N)] = Y

    
if __name__ == "__main__":
    with shelve.open('shelve/perform_numba-parallel.shelve') as shelf:
        shelf.clear()

    for N in [1,10,100,1_000,2_500,5_000,10_000]: 
        for i in range(0,4):
            main(N) 