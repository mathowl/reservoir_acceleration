import numpy as np #1.27 because 1.3 is not compatible with numba

import torch
import time
import math
import shelve


def Df_gpu(m,Wcp,P,v,N):
    b=torch.zeros((N,3)).to('cuda')
    b[:,2] =  (P[7]  + P[2] * m[:,2])
    b[:,0] = P[3] * torch.matmul(Wcp,m[:,0])
    b-= torch.multiply(torch.linalg.cross( m, v), \
                    ((P[5])/(1+P[6] * torch.matmul(m, v[0,:]) )).reshape(-1,1))    
    c=torch.linalg.cross(m, b )
    return (-1 * c *P[1] - (P[0]) *P[1]  * torch.linalg.cross(m , c)) 





def frk4_gpu_step(y,Wcp,h,P,v,N):
    k1 = h * Df_gpu( y,Wcp,P,v,N)
    k2 = h * Df_gpu(y + 0.5 * k1,Wcp,P,v,N)
    k3 = h * Df_gpu( y + 0.5 * k2,Wcp,P,v,N)
    k4 = h * Df_gpu(y + k3,Wcp,P,v,N)
    y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    return y




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





def Minitial(N):
    
    return np.array([list(sphericalCoordinateToCartesianCoordinate([1,2*math.pi/360,2*math.pi/360]))]*N)

def sphericalCoordinateToCartesianCoordinate(sphiricalCoordinate):
    return sphiricalCoordinate[0]*np.array([math.sin(sphiricalCoordinate[1])*math.cos(sphiricalCoordinate[2])
                                          , math.sin(sphiricalCoordinate[1])*math.sin(sphiricalCoordinate[2])
                                          , math.cos(sphiricalCoordinate[1])])



def main(N):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.float64)

    with torch.no_grad():
        P = np.array([0.005, \
            1/(1+0.005**2),\
            (18.616e3)   - 4*np.pi*1448.3,\
            1,\
            8500,\
            1.05457266e9*2.5*0.537/(2*1.60217733*720*np.pi*1448.3), \
            0.288,\
            200,\
            0,\
            416.12543922361147
            ])
        P0 = np.array([0.005, \
            1/(1+0.005**2),\
            ((18.616e3)   - 4*np.pi*1448.3)/P[9],\
            1/P[9],\
            8500,\
            1.05457266e9*2.5*0.537/(2*1.60217733*720*np.pi*1448.3*P[9]), \
            0.288,\
            200/P[9],\
            416.12543922361147])

        nsteps=5000
        h =  17640000.0 * 1e-11 * P[9]

        M0 = Minitial(N)

    
        np.random.seed(0)

        Wcp0 =  2*np.random.rand(N,N)-1
        if N !=1:
            for i in range(N):
                Wcp0[i,i] = 0
            eigenValues,__ = np.linalg.eig(Wcp0)
            Wcp0 *= 1/np.amax(np.abs(eigenValues))
        else:
            Wcp0 = np.array([[0]])


        Wcp0= torch.from_numpy(Wcp0).to('cuda') 
        M0 = torch.from_numpy(M0).to('cuda')
        P0 = torch.from_numpy(P0).to('cuda')
        
        v_dummy= torch.tensor([1.000000e+00, 0.000000e+00, 6.123234e-17])
        v= v_dummy.repeat(N,1).to('cuda')

        
        
        
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        
        Msol = np.zeros((3,nsteps))
        s0 =0
        Msol[:,0]=M0[0,:].data.cpu().numpy() 
        for i in range(0,nsteps-1): 
            
            start.record()
            
            
            M0 = frk4_gpu_step(M0,Wcp0,h,P0,v,N)
            Msol[:,i+1] = M0[0,:].data.cpu().detach().numpy()
            end.record()
             
            torch.cuda.synchronize()
            
            s0 += start.elapsed_time(end)
        
        print(s0/1000)
        with shelve.open('shelve/perform_gpu.shelve', 'c') as shelf:
            shelf[str(time.time())] = [N,s0/1000]

        with shelve.open('shelve/sol_gpu.shelve', 'c') as shelf:
            shelf[str(N)] = Msol.T

    
if __name__ == "__main__":

    print(torch.cuda.get_device_name())
    print(torch.__version__)
    print(torch.version.cuda)
    x = torch.randn(1).cuda()
    print(x)

    for _ in range(10): #warm-up
        main(1000)
        torch.cuda.empty_cache()

    with shelve.open('shelve/perform_gpu.shelve') as shelf:
        shelf.clear()
    time.sleep(1)

    for N in [10,100,1_000,2_500,5_000,10_000]: 
        for i in range(0,4):
            torch.cuda.empty_cache()
            main(N) 