import numpy as np
from numba import jit
import copy
import pickle



@jit(nopython=True)
def makesticky(theta,x): # see appendix D3 of micro jumps macro humps paper

    xsticky=x*0

    xsticky[:,0]=x[:,0]    
    xsticky[0,1:x.shape[1]]=(1-theta)*x[0,1:x.shape[1]]    

    for t in range(1,x.shape[0]):
        for s in range(1,x.shape[1]):

            xsticky[t,s]=theta*xsticky[t-1,s-1]+(1-theta)*x[t,s]

    return xsticky 


def stick_jacob(J,theta):

    Jsticky=copy.deepcopy(J)

    for i in J.outputs:

        for j in J.inputs:
            
            x=J[i][j]
            
            xsticky=makesticky(theta,x)
            Jsticky[i][j]=xsticky

    return Jsticky

