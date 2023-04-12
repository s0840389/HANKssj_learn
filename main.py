import os
import pickle

mainpath='/home/jamie/OneDrive/Documents/research/HANK/HANKssj'

os.chdir(mainpath)

import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import simple, solved, combine, create_model  # functions
#from sequence_jacobian import grids, hetblocks                       # modules

from sequence_jacobian import grids                       # modules
from sticky_exp import *

saveobj=True
loadss=True

## blocks

# philips curve

@solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1/mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / \
           (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc

# equity price [ arbritrage condition]

@solved(unknowns={'p': (5, 15)}, targets=['equity'], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


# Aggregate labour demand and marginal costs [Firm FOC]

@simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc

# Aggregate investment [Q equation from adjustment costs] 

@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) -\
        (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) +\
        K(+1) / K * Q(+1) - (1 + r(+1)) * Q
    return inv, val


production = combine([labor, investment])                              # create combined block
production_solved = production.solved(unknowns={'Q': 1., 'K': 10.},    # turn it into solved block
                                      targets=['inv', 'val'],
                                      solver='broyden_custom')

# dividend as residual from income

@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - w * N - I - psip
    return psip, I, div

# taylor rule [exogenous rstar]

@simple
def taylor(rstar, pi, phi):
    i = rstar + phi * pi
    return i

# fiscal (balanced budget)

@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax


# real interest rate and return on assets 
@simple
def finance(i, pi, r ,omega):
    rb = r - omega
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, fisher


# wage inflation 

@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw

# wage philips curve

@simple
def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw) + beta * \
            (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log)
    return wnkpc

# market clearing

@simple
def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + psip + omega * B - Y
    return asset_mkt, wealth, goods_mkt


# equity share of A

@simple
def share_value(p, wealth, Bh):
    pshare = p / (wealth - Bh)
    return pshare

@simple
def rareturn(pshare,div,r,ra,p):
    checkra= pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1 -ra
    return checkra
    

# Household problem

def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid

# off the shelf input from auclert et al but can define own household function

from hh_prob import *

#hh = hetblocks.hh_twoasset.hh
hh_ext = hh.add_hetinputs([income, make_grids])



#@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
#     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
#def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1):
   #................
   #.................
   #return Va, Vb, a, b, c, uce


# define steady state

@simple
def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
    """Solves for (mup, alpha, Z, w) to hit (tot_wealth, Y, K, pi)."""
    # 1. Solve for markup to hit total wealth
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc

    # 2. Solve for capital share to hit K
    alpha = (r + delta) * K / Y / mc

    # 3. Solve for TFP to hit Y
    Z = Y * K ** (-alpha) * N ** (alpha - 1)

    # 4. Solve for w such that piw = 0
    w = mc * (1 - alpha) * Y / N

    return p, mc, mup, alpha, Z, w


@simple
def union_ss(tax, w, UCE, N, muw, frisch):
    """Solves for (vphi) to hit (wnkpc)."""
    vphi = (1 - tax) * w * UCE / muw / N ** (1 + 1 / frisch)
    wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * UCE / muw
    return vphi, wnkpc

blocks_ss = [hh_ext, partial_ss, union_ss, dividend, taylor, fiscal, share_value, finance, mkt_clearing,rareturn]

hank_ss = create_model(blocks_ss, name='Two-Asset HANK SS')

# calibrate and solve for steady state

if loadss:
    
    pickle_in = open("out_data/HANK.pickle","rb")
    cali = pickle.load(pickle_in)[2]

else:
    calibration = {'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125, 'tot_wealth': 14,
               'delta': 0.02, 'pi': 0., 'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8,
               'G': 0.2, 'eis': 0.5, 'frisch': 1, 'chi0': 0.25, 'chi2':2 , 'epsI': 4,
               'omega': 0.005, 'kappaw': 0.1, 'phi': 1.5, 'nZ': 5, 'nB': 50, 'nA': 70, 'nK': 50,
               'bmax': 40, 'amax': 3000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92}

    unknowns_ss = {'beta': 0.976, 'chi1': 10.5,'ra':0.0125}
    targets_ss = {'asset_mkt': 0., 'B': 'Bh','checkra':0}

    cali = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='broyden_custom')


# define dynamics blocks and model

blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved,
          dividend, taylor, fiscal, share_value,
          finance, wage, union, mkt_clearing,rareturn]

hank = create_model(blocks, name='Two-Asset HANK')

#ss =  hank.steady_state(cali)
ss=cali

# impulse responses

# MIT shock

exogenous = ['rstar', 'Z', 'G'] # exogenous variables
unknowns = ['r', 'w', 'Y','ra'] # unknown variables
targets = ['asset_mkt', 'fisher', 'wnkpc','checkra'] # equations to pin down
T = 300

drstar = -0.0025 * 0.8 ** (np.arange(T)[:, np.newaxis])

drstar_fwd=np.roll(drstar,10)
drstar_fwd[:10]=0


#td_nonlin = hank.solve_impulse_nonlinear(ss, unknowns, targets, {"rstar": drstar[:,0]},)

# sequence space Jacobian soln [ can give some jacobians for some blocks]

J_ha = hh_ext.jacobian(ss, inputs=['N', 'r', 'ra', 'rb', 'tax', 'w'], T=T)

G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha})


J_ha_sticky=stick_jacob(J_ha,0.94) # Reduce forwarding lookingness of hh Jacobian and resolve model

G_sticky = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha_sticky})


# Consumption decomposition chart

dc_lin=100*G['C']['rstar'] @ drstar/ss['C']

fig,ax =plt.subplots()

tt=np.arange(0,T)

yyoldminus=0*tt[0:24]
yyoldplus=0*yyoldminus

bcolor=['darkblue','darkgreen','orange','grey','gold']

iter=0
for i in ['rb','ra','w','tax','N']:
    
    yy=J_ha['C'][i]@G[i]['rstar']@drstar/ss['C']*100 # combine HH jacobian with GE inputs 

    ax.bar(tt[:24],yy[:24,-1].clip(min=0),bottom=yyoldplus,label=i,color=bcolor[iter])
    ax.bar(tt[:24],yy[:24,-1].clip(max=0),bottom=yyoldminus,color=bcolor[iter])
    
    yyoldplus=yy[:24,-1].clip(min=0)+yyoldplus
    yyoldminus=yy[:24,-1].clip(max=0)+yyoldminus

    iter=iter+1

plt.plot(dc_lin[:24], label='Total', linestyle='-', linewidth=2.5)

ax.legend()
plt.title('Decomposition of consumption response to monetary policy (FI)')
plt.show()


# Output IRFs comparison


varpick='Y'
shockpick='Z'
shockname='TFP shock'

dy_lin=100*G[varpick][shockpick] @ drstar/ss['Y']
dy_lin_sticky=100*G_sticky[varpick][shockpick] @ drstar/ss['Y']
dy_lin_fwd=100*G[varpick][shockpick] @ drstar_fwd/ss['Y']

fig2,ax=plt.subplots()

plt.plot(dy_lin[:24], label='linear', linestyle='-', linewidth=2.5)
plt.plot(dy_lin_fwd[:24], label='linear forward', linestyle='--', linewidth=2.5)
plt.plot(dy_lin_sticky[:24], label='Linear sticky', linestyle='--', linewidth=2.5)

tit=varpick+' response to 1% '+shockname

plt.title(tit)
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()


## steady state distributions

Bdist=np.sum(np.sum(ss.internals['hh']['D'],axis=0),axis=1)
bgrid=ss.internals['hh']['b_grid']


fig3, ax=plt.subplots()

ax.bar(bgrid,Bdist)
plt.title('Liquid asset distribution')
plt.show()


Adist=np.sum(np.sum(ss.internals['hh']['D'],axis=0),axis=0)
agrid=ss.internals['hh']['a_grid']

fig4, ax=plt.subplots()

ax.bar(agrid,Adist)
plt.title('Iliquid asset distribution')
plt.show()


Zdist=np.sum(np.sum(ss.internals['hh']['D'],axis=1),axis=1)
Zgrid=ss.internals['hh']['e_grid']

fig5, ax=plt.subplots()

ax.bar(Zgrid,Zdist)
plt.title('Income distribution')
plt.show()

htm_share=ss['SHRHTM']
htm_consump=ss['CHTM']/ss['SHRHTM']

## other IRFs

Airf=G['A']['rstar']@drstar+ss['A']
Birf=G['B']['rstar']@drstar+ss['B']
Wealthirf=G['B']['rstar']@drstar+ss['wealth']
Pirf=G['p']['rstar']@drstar+ss['p']
Pshareirf=G['pshare']['rstar']@drstar+ss['pshare']


# save objects

if saveobj:
    pickle_out = open("out_data/HANK.pickle","wb")
    pickle.dump([G,J_ha,ss], pickle_out)
    pickle_out.close()




