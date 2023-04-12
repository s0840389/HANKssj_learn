import copy
import numpy as np
import matplotlib.pyplot as plt

from numba import jit

from sequence_jacobian import het, simple, create_model              # functions
from sequence_jacobian import interpolate, grids, misc, estimation   # modules



## household problem

def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va

@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, e_grid, r, w, beta, eis):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Parameters
    ----------
    Va_p     : array (nE, nA), expected marginal value of assets next period
    a_grid   : array (nA), asset grid
    e_grid   : array (nE), producticity grid
    r        : scalar, ex-post real interest rate
    w        : scalar, wage
    beta     : scalar, discount factor
    eis      : scalar, elasticity of intertemporal substitution

    Returns
    ----------
    Va : array (nE, nA), marginal value of assets today
    a  : array (nE, nA), asset policy today
    c  : array (nE, nA), consumption policy today
    """
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid


household_ext = household.add_hetinputs([make_grid])


## firms and market clearing

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt


# model 

ks = create_model([household_ext, firm, mkt_clearing], name="Krusell-Smith")

# steady state
calibration = {'eis': 1, 'delta': 0.025, 'alpha': 0.11, 'rho_e': 0.966, 'sd_e': 0.5, 'L': 1.0,
               'nE': 7, 'nA': 500, 'amin': 0, 'amax': 200}
unknowns_ss = {'beta': 0.98, 'Z': 0.85, 'K': 3.}
targets_ss = {'r': 0.01, 'Y': 1., 'asset_mkt': 0.}

ss = ks.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='hybr')

# dynamics
T=200

J_ha = household.jacobian(ss, inputs=['r', 'w'], T=T) # full info jacobian

# sticky expectations
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


J_ha_sticky=stick_jacob(J_ha,0.94) # sticky jacobian

inputs = ['Z']
unknowns = ['K']
targets = ['asset_mkt']

G = ks.solve_jacobian(ss, unknowns, targets, inputs, T=T,Js={'household':J_ha})
G_sticky = ks.solve_jacobian(ss, unknowns, targets, inputs, T=T,Js={'household':J_ha_sticky})


dZ = 100*0.01 * 0.8 ** (np.arange(T)[:, np.newaxis]) #tfp shock


dc = G['C']['Z'] @ dZ
dc_sticky = G_sticky['C']['Z'] @ dZ

plt.figure(1)

plt.plot(dc[:21], label='full info', linestyle='-', linewidth=2.5)
plt.plot(dc_sticky[:21], label='sticky', linestyle='--', linewidth=2.5)
tit=' Consumption response to 1% TFP shock'
plt.title(tit)
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()