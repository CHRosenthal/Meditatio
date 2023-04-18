'''
Systems bio - bacterial warfare
Author: C.H. Rosenthal
Note that commented section takes hours to implement !!!
'''

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

rng = np.random.Generator(np.random.MT19937())

def sim_battle_cont(alpha, pm, pn, tmax = 10, m0 = 100, n0 = 100,
                    ax = None, nodisplay = False):
  pop_init = [m0,n0]
  def mdl(t,x):
    d = []
    d.append(pm[2]*alpha*x[0]**2/sum(x) - pn[0]*x[0]*x[1]/pm[1]/sum(x))
    d.append(pn[2]*alpha*x[1]**2/sum(x) - pm[0]*x[1]*x[0]/pn[1]/sum(x)) 
    return d
  res = sp.integrate.solve_ivp(mdl,(0,tmax),pop_init,'LSODA')
  if not nodisplay:
    if type(ax) == type(None):
      _, ax = plt.subplots(figsize=(6,5))
    ax.semilogy(res.t, res.y[0,:],label=f'T{pm[0]:.2}, R{pm[1]:.2}, P{pm[2]:.2}')
    ax.semilogy(res.t, res.y[1,:],label=f'T{pn[0]:.2}, R{pn[1]:.2}, P{pn[2]:.2}')
    ax.legend()
    ax.set_title(f'alpha = {alpha}')
  win = res.y[0,-1]>res.y[1,-1]
  return win, res

sim_battle_cont(7,[1/3,1/3,1/3],[1/4,1/4,1/2])
sim_battle_cont(3,[1/3,1/3,1/3],[1/4,1/4,1/2])

def sim_battle_disc(alpha, pm, pn, tmax = 10, m0 = 100, n0 = 100,
                    ax = None, nodisplay = False):
  t = [0]
  now = 0
  m = [m0]
  m_now = m0
  n = [n0]
  n_now = n0
  while now < tmax and len(n) < 10**4:
    pmadd = pm[2]*alpha * m_now**2 / (m_now+n_now)
    pmred = pn[0]/pm[1] * m_now * n_now / (m_now+n_now)
    pnadd = pn[2]*alpha * n_now **2/ (m_now+n_now)
    pnred = pm[0]/pn[1] * n_now * m_now / (m_now+n_now)
    sump = pmadd+pmred+pnadd+pnred
    now += rng.exponential(1/sump)
    t.append(now)
    tmp = rng.random()
    if tmp < pmadd/sump:
      m_now += 1; m.append(m_now); n.append(n_now)
    elif tmp < (pmred+pmadd)/sump:
      m_now -= 1; m.append(m_now); n.append(n_now)
    elif tmp < (pmred+pmadd+pnadd)/sump:
      n_now += 1; m.append(m_now); n.append(n_now)
    else:
      n_now -= 1; m.append(m_now); n.append(n_now)
    if n_now == 0 or m_now == 0: break
  win = m_now>n_now
  ts = pd.DataFrame(dict(t=t,m=m,n=n))
  if not nodisplay:
    if type(ax) == type(None):
      _, ax = plt.subplots(figsize=(6,5))
    ax.semilogy(t, m,label=f'T{pm[0]:.2}, R{pm[1]:.2}, P{pm[2]:.2}')
    ax.semilogy(t, n,label=f'T{pn[0]:.2}, R{pn[1]:.2}, P{pn[2]:.2}')
    ax.legend()
    ax.set_title(f'alpha = {alpha}')
  return win, ts

fig, ax = plt.subplots(1,3,figsize=(17,5))
sim_battle_cont(3,[0.6,0.2,0.2],[1/4,1/4,1/2], ax = ax[0])
sim_battle_cont(3,[1/4,1/2,1/4],[1/4,1/4,1/2], ax = ax[1])
sim_battle_cont(3,[0.6,0.2,0.2],[1/4,1/2,1/4], ax = ax[2])
sim_battle_disc(3,[0.6,0.2,0.2],[1/4,1/4,1/2], ax = ax[0])
sim_battle_disc(3,[1/4,1/2,1/4],[1/4,1/4,1/2], ax = ax[1])
sim_battle_disc(3,[0.6,0.2,0.2],[1/4,1/2,1/4], ax = ax[2])

fig, ax = plt.subplots(1,3,figsize=(17,5))
sim_battle_cont(7,[0.6,0.2,0.2],[1/4,1/4,1/2], ax = ax[0])
sim_battle_cont(7,[1/4,1/2,1/4],[1/4,1/4,1/2], ax = ax[1])
sim_battle_cont(7,[0.6,0.2,0.2],[1/4,1/2,1/4], ax = ax[2])
sim_battle_disc(7,[0.6,0.2,0.2],[1/4,1/4,1/2], ax = ax[0])
sim_battle_disc(7,[1/4,1/2,1/4],[1/4,1/4,1/2], ax = ax[1])
sim_battle_disc(7,[0.6,0.2,0.2],[1/4,1/2,1/4], ax = ax[2])

pm_test = rng.dirichlet(alpha = (1,1,1))
pn_test = rng.dirichlet(alpha = (1,1,1))
for i in range(1,10):sim_battle_disc(i,pm_test, pn_test)

#simsize = 100
#p_sample = rng.dirichlet(alpha = (1,1,1), size = simsize)

res = 0.05
p_sample = []
for i in np.arange(res,1-res,res):
  for j in np.arange(res,1-i,res):
    p_sample.append([i,j,1-i-j])
simsize = len(p_sample)
p_sample = np.array(p_sample)

description = []
for i in range(simsize):
  tmp = p_sample[i,:]
  if tmp[0] > 0.5: description.append('T')
  elif tmp[1] > 0.5: description.append('R')
  elif tmp[2] > 0.5: description.append('S')
  else: description.append('B')
description = np.array(description, dtype = '<U1')
pinfo = pd.DataFrame(dict(toxin = p_sample[:,0],resistance = p_sample[:,1],
    prolif = p_sample[:,2],description = description),
    index = [f'{description[k]}{p_sample[k,0]:.2f},{p_sample[k,1]:.2f},{p_sample[k,2]:.2f}'
             for k in range(simsize)]).sort_values(by=['description','resistance'])
alist = [10**(-4),.1,.2,.5,1,2,3,4,5,10,20,50]
# mat = np.zeros(shape = (simsize,simsize,len(alist))).astype('?')
# tic = time.perf_counter()
# for i in range(simsize):
#   pm = pinfo.iloc[i,:3].values
#   for j in range(i+1, simsize):
#     pn = pinfo.iloc[j,:3].values
#     for k in range(len(alist)):
#       alpha = alist[k]
#       win,_ = sim_battle_cont(alpha, pm, pn, nodisplay = True)
#       mat[i,j,k] = win
#       mat[j,i,k] = ~win
#   toc = time.perf_counter() - tic
#   print(f'Iterating over strain {i}, time = {toc:i} seconds')


# for k in range(len(alist)):
#   sns.heatmap(mat[:,:,k])
#   plt.show()
#   pinfo[f'winrate_{alist[k]}'] = mat[:,:,k].mean(axis=1)
#   sns.scatterplot(pinfo,x = 'prolif',y = 'toxin', hue = f'winrate_{alist[k]}',
#                   ax = plt.subplots(figsize = (10.5,10))[1])
#   plt.show()
# sns.heatmap(pinfo.iloc[:,:3])

def sim_battle3_disc(alpha, pm, pn, pq, tmax = 10, m0 = 100, n0 = 100, q0 = 100,
                    ax = None, nodisplay = False):
  t = [0]
  now = 0
  m = [m0]
  m_now = m0
  n = [n0]
  n_now = n0
  q = [q0]
  q_now = q0
  while now < tmax and len(n) < 10**4:
    pmadd = pm[2]*alpha * m_now**2
    pmred = pn[0]/pm[1] * m_now * n_now + pq[0]/pm[1] * m_now*q_now
    pnadd = pn[2]*alpha * n_now **2
    pnred = pm[0]/pn[1] * n_now * m_now + pq[0]/pn[1] * n_now*q_now
    pqadd = pq[2] * alpha * q_now **2
    pqred = pm[0]/pq[1] * m_now * q_now + pn[0]/pq[1] * m_now*q_now
    pmred *=.5; pnred *=.5; pqred *=.5
    sump = pmadd+pmred+pnadd+pnred+pqadd+pqred
    now += rng.exponential((m_now+n_now+q_now)/sump)
    t.append(now)
    tmp = rng.random()
    if tmp < pmadd/sump:
      m_now += 1; m.append(m_now); n.append(n_now); q.append(q_now)
    elif tmp < (pmred+pmadd)/sump:
      m_now -= 1; m.append(m_now); n.append(n_now); q.append(q_now)
    elif tmp < (pmred+pmadd+pnadd)/sump:
      n_now += 1; m.append(m_now); n.append(n_now); q.append(q_now)
    elif tmp < (pmred+pmadd+pnadd+pnred)/sump:
      n_now -= 1; m.append(m_now); n.append(n_now); q.append(q_now)
    elif tmp < 1-pqred/sump:
      q_now += 1; m.append(m_now); n.append(n_now); q.append(q_now)
    else:
      q_now -= 1; m.append(m_now); n.append(n_now); q.append(q_now)
  ts = pd.DataFrame(dict(t=t,m=m,n=n,q=q))
  if not nodisplay:
    if type(ax) == type(None):
      _, ax = plt.subplots(figsize=(6,5))
    ax.semilogy(t, m,label=f'T{pm[0]:.2}, R{pm[1]:.2}, P{pm[2]:.2}')
    ax.semilogy(t, n,label=f'T{pn[0]:.2}, R{pn[1]:.2}, P{pn[2]:.2}')
    ax.semilogy(t, q,label=f'T{pq[0]:.2}, R{pq[1]:.2}, P{pq[2]:.2}')
    ax.legend()
    ax.set_title(f'alpha = {alpha}')
  return ts

for alpha in alist:
  sim_battle3_disc(alpha, [0.6,0.2,0.2],[1/4,1/2,1/4],[1/4,1/4,1/2])

def sim_battle3_cont(alpha, pm, pn, pq, tmax = 10, m0 = 100, n0 = 100, q0 = 100,
                    ax = None, nodisplay = False):
  pop_init = [m0,n0,q0]
  def mdl(t,x):
    d = []
    d.append(pm[2]*alpha*x[0]**2/sum(x) - .5*(pn[0]*x[1]+pq[0]*x[2])*x[0]/pm[1]/sum(x))
    d.append(pn[2]*alpha*x[1]**2/sum(x) - .5*(pm[0]*x[0]+pq[0]*x[2])*x[1]/pn[1]/sum(x))
    d.append(pq[2]*alpha*x[2]**2/sum(x) - .5*(pm[0]*x[0]+pn[0]*x[1])*x[2]/pq[1]/sum(x))
    return d
  res = sp.integrate.solve_ivp(mdl,(0,tmax),pop_init,'LSODA')
  if not nodisplay:
    if type(ax) == type(None):
      _, ax = plt.subplots(figsize=(6,5))
    ax.semilogy(res.t, res.y[0,:],label=f'T{pm[0]:.2}, R{pm[1]:.2}, P{pm[2]:.2}')
    ax.semilogy(res.t, res.y[1,:],label=f'T{pn[0]:.2}, R{pn[1]:.2}, P{pn[2]:.2}')
    ax.semilogy(res.t, res.y[2,:],label=f'T{pq[0]:.2}, R{pq[1]:.2}, P{pq[2]:.2}')
    ax.legend()
    ax.set_title(f'alpha = {alpha}')
  return

fig,ax = plt.subplots(1,len(alist),figsize = (5*len(alist)+1,5))
for i in range(len(alist)):
  alpha = alist[i]
  sim_battle3_cont(alpha, [0.6,0.2,0.2],[1/4,1/2,1/4],[1/4,1/4,1/2],ax = ax[i],tmax=6)
  sim_battle3_disc(alpha, [0.6,0.2,0.2],[1/4,1/2,1/4],[1/4,1/4,1/2],ax = ax[i])
  
p1 = rng.dirichlet((1,1,1)); p2 = rng.dirichlet((1,1,1)); p3 = rng.dirichlet((1,1,1))
fig,ax = plt.subplots(1,len(alist),figsize = (5*len(alist)+1,5))
for i in range(len(alist)):
  alpha = alist[i]
  sim_battle3_cont(alpha, p1,p2,p3,ax = ax[i],tmax=6)
  sim_battle3_disc(alpha, p1,p2,p3,ax = ax[i])
