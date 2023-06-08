'''
Systems bio - GK Special Edition
Author: C.H. Rosenthal
'''

import numpy as np
import scipy as sp
import scipy.stats as sts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import time

# The red-blue colour map
cdict = dict(red = ((0,1,1),(1/6,0,0),(1/2,0,0),(2/3,1,1),(1,1,1)),
             green = ((0,1,1),(1/6,1,1),(1/3,0,0),(2/3,0,0),(5/6,1,1),(1,1,1)),
             blue = ((0,1,1),(1/3,1,1),(1/2,0,0),(5/6,0,0),(1,1,1)))
cmap_name = 'bipolar'
cmap = mpl.colors.LinearSegmentedColormap(cmap_name,cdict,1024)
try:
  mpl.colormaps.register(cmap) # if running repeatedly, this will raise an error
except:
  a=1 # just a filler code that does not do anything

# Part A - Generate random field (distribution of resources in a 2D plane)
# A field is limited to -3 < x < 3, -3 < y < 3 and consists of 10+
# normally distributed clusters (variance 1) summed together
# Let total resources be 1

rng = np.random.Generator(np.random.MT19937(seed = 114514))

def generate_f():
  # We randomly generate fields to enhance generalisability of strategies
  n = rng.poisson(5) + 10 # mean number of peaks = 15
  cluster_heights = rng.dirichlet(np.ones(n)*3)*n # mean height = 1
  cluster_coords = rng.uniform(-4,4,size = 2*n).reshape((n,2))
  def f(x,y): # this function is returned
    z = 0
    for i in range(n):
      d = (x-cluster_coords[i,0])**2 + (y-cluster_coords[i,1])**2
      d = d ** 0.5 # distance to cluster for bivariate normal pdf
      z += sts.norm.pdf(d) * cluster_heights[i]
    return z # this concludes the field function
  return f # this returns the function handle

# Example field
def plotf(f, ax = None, show = True):
  X, Y = np.meshgrid(np.linspace(-6,6,1024), np.linspace(-6,6,1024))
  z = f(X,Y)
  if type(ax) == type(None): _,ax = plt.subplots(figsize = (7,7))
  ax.contourf(X,Y,z, levels = 20, cmap = 'hot')
  if show: plt.show()
  return
f = generate_f()
plotf(f)

# Part B - Random walk for a single individual

f = generate_f()

# We denote the bias parameter alpha, lower alpha = higher tendency to explore
def sim_individual(alpha,
                   f = generate_f(), 
                   x_init = np.random.uniform(-4,4,size=2), # initial coordinates
                   max_iter = 1000, step = 0.05, quiet = True, ax = None):
  if alpha < 0: raise ValueError
  x_list = x_init.copy() # list of coordinates across 100 iterations
  x_now = x_init.copy()
  score = f(x_now[0], x_now[1])
  score_now = f(x_now[0],x_now[1])
  
  for _ in range(max_iter - 1):
    x_cand = rng.multivariate_normal(mean = x_now,cov = np.diag([step,step]))
    score_cand = f(x_cand[0],x_cand[1])
    prob = sp.special.expit((score_cand-score_now)*alpha)
    # the probability of accepting the candidate location is given by a sigmoid
    # function of the score difference, lower alpha = more likely to accept a 
    # decrease in score, less likely to accept an increase
    
    if rng.random() < prob: # move to the next location
      score_now = score_cand
      x_now = x_cand
    
    x_list = np.row_stack((x_list, x_now))
    score += score_now 
    # The score (total fitness) equals the sum of resources at each time point
  
  if not quiet:
    if type(ax) == type(None): _,ax = plt.subplots(figsize = (7,7))
    X,Y = np.meshgrid(np.linspace(-6,6,1024), np.linspace(-6,6,1024))
    z = f(X,Y)
    ax.contourf(X,Y,z, levels = 20, cmap = 'hot')
    l = mpl.lines.Line2D(x_list[:,0], x_list[:,1],c = 'b')
    ax.add_line(l)
    plt.xlim([-6,6]); plt.ylim([-6,6])
    plt.show()
    print(f'Score = {score:.3f}, final score = {score_now:.3f}')
    
  return score, score_now, x_list

tic = time.perf_counter()
score = sim_individual(25,f,quiet = False)
toc = time.perf_counter()-tic

# we simulate 1000 points for the same field and same starting location
n_trials = 1000
alpha_list = np.linspace(1,100,n_trials)
_,ax = plt.subplots(1,3,figsize = (22,7))
plotf(f,ax[2], show = False)
x_init = rng.uniform(-4,4,2)
i = 0
tic = time.perf_counter()
score_list = []
final_list = []
x_init = rng.uniform(-4,4,2)
ax[2].scatter(x_init[0],x_init[1])
for a in alpha_list:
  score, final, _ = sim_individual(a,f, x_init, max_iter = 500)
  score_list.append(score)
  final_list.append(final)
  i += 1
  if i % 50 == 0:
    toc = time.perf_counter()-tic
    print(f'{i}/{n_trials*5*5} completed, time = {toc:.3f} seconds')
# spl = sp.interpolate.UnivariateSpline(alpha_list,score_list)

ax[0].scatter(alpha_list,score_list, marker = '.', s = 25)
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('score')
ax[1].scatter(alpha_list,final_list, marker = '.', s = 25)
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('score')


# for 5 different fields we do 64 x 5 x 5 experiments from the same starting position
n_trials = 64
alpha_list = np.linspace(10,255,n_trials)
_,ax = plt.subplots(1,2,figsize = (14,7))
alphas = []
scores = []
finals = []
init_list = []

i = 0
tic = time.perf_counter()
for _ in range(5):
  f = generate_f() # 5 independent fields
  for _ in range(5): # 5 independent starting locations
    score_list = []
    final_list = []
    x_init = rng.uniform(-4,4,2)
    init_list.append(x_init)
    for a in alpha_list:
      score, final, _ = sim_individual(a,f, x_init, max_iter = 500)
      score_list.append(score)
      final_list.append(final)
      i += 1
      if i % 50 == 0:
        toc = time.perf_counter()-tic
        print(f'{i}/{n_trials*5*5} completed, time = {toc:.3f} seconds')
    # spl = sp.interpolate.UnivariateSpline(alpha_list,score_list)
    scores.append(score_list)
    finals.append(final_list)
    alphas.append(alpha_list)
  
    ax[0].scatter(alpha_list,score_list, marker = '.', s = 25)
    ax[0].set_xlabel('alpha')
    ax[0].set_ylabel('score')
    ax[1].scatter(alpha_list,final_list, marker = '.', s = 25)
    ax[1].set_xlabel('alpha')
    ax[1].set_ylabel('score')
  
plt.show()

scores = np.array(scores).reshape(25*n_trials)
finals = np.array(finals).reshape(25*n_trials)
finals_bin = finals >= 0.9 * finals.max() # This identifies those reaching global maximum
alphas = np.array(alphas).reshape(25*n_trials)
dat = pd.DataFrame(dict(alpha = alphas, final_bin = finals_bin, final = finals,
                        score = scores))
dat['log_alpha'] = np.log(dat.alpha)
dat['alpha_sq'] = dat.alpha ** 2
dat['alpha_inv'] = dat.alpha ** (-1)
mdl = sm.formula.glm(formula = 'final_bin ~ alpha_inv', data = dat,
             family = sm.families.Binomial()).fit()
sns.lmplot(data = dat, x = 'alpha_inv', y = 'final_bin', logistic = True)

# Part C - Competitive random walk
def sim_competition(alpha, # size 2
                   f = generate_f(),
                   beta = 0.3, # competition parameter
                   x_init = np.random.uniform(-4,4,size=2), # initial coordinates
                   max_iter = 1000, movement_rate = 0.05, quiet = True, ax = None):
  if alpha[0] < 0 or alpha[1] < 0: raise ValueError
  if len(x_init) == 2: x_init = x_init.repeat(2)[[0,2,1,3]]
  x_list = x_init.copy() # list of coordinates across 100 iterations
  x_now = x_init.copy()
  d_now = (x_init[0]-x_init[2])**2 + (x_init[1]-x_init[3])**2
  d_now = d_now ** 0.5
  score = np.array([f(x_now[0], x_now[1]), f(x_now[2],x_now[3])]) # 2 scores
  score_now = score
  score_now[0] *= sts.norm.pdf(0, scale = beta) / \
    (sts.norm.pdf(0, scale = beta) + \
     np.exp(score[1]-score[0]) * sts.norm.pdf(d_now, scale = beta))
  score_now[1] *= sts.norm.pdf(0, scale = beta) / \
    (sts.norm.pdf(0, scale = beta) + \
     np.exp(score[0]-score[1]) * sts.norm.pdf(d_now, scale = beta))
  # In the competition, we assume the population of each strain of bacteria to 
  # be exponentially growing with the score per step, and they spread normally 
  # by sd = beta,  
  # and accordingly occupy the resources of the other strain, thus multiplying
  # the score above.
  # We use log population as the score
  
  score = score_now # we re-set the scores
  score_ts = score
  step_ts = score
  
  # we make the two items to take turns in determining the new location
  for _ in range(max_iter - 1):
    x_cand_a = rng.multivariate_normal(mean = x_now[0:2],cov = np.diag([movement_rate]*2))
    d_cand_a = (x_cand_a[0]-x_now[2])**2 + (x_cand_a[1]-x_now[3])**2
    score_cand_a = f(x_cand_a[0],x_cand_a[1])
    score_cand_a *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       np.exp(score[1]-score[0]) * sts.norm.pdf(d_cand_a, scale = beta))
    score_stay_a = f(x_now[0],x_now[1])
    score_stay_a *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       np.exp(score[1]-score[0]) * sts.norm.pdf(d_now, scale = beta))
    prob_a = sp.special.expit((score_cand_a-score_stay_a)*alpha[0])
    # when evaluating the occupation of resources by B, we assume B to stay
    # this is a heuristic to assist computation
    # since we need to apply the same heuristic to B (to be fair)
    # we update the values later
        
    x_cand_b = rng.multivariate_normal(mean = x_now[2:4],cov = np.diag([movement_rate]*2))
    d_cand_b = (x_cand_b[0]-x_now[0])**2 + (x_cand_b[1]-x_now[1])**2
    score_cand_b = f(x_cand_b[0],x_cand_b[1])
    score_cand_b *=  sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       np.exp(score[0]-score[1]) * sts.norm.pdf(d_cand_b, scale = beta))
    score_stay_b = f(x_now[2],x_now[3])
    score_stay_b *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       np.exp(score[0]-score[1]) * sts.norm.pdf(d_now, scale = beta))
    prob_b = sp.special.expit((score_cand_b-score_stay_b)*alpha[1])
    
    step = []
    if rng.random() < prob_a: # move to the next location (a/b independently)
      score[0] += score_cand_a
      step.append(score_cand_a)
      x_now[0:2] = x_cand_a
    else: # we do not update x_now
      score[0] += score_stay_a
      step.append(score_stay_a)
    
    if rng.random() < prob_b:
      score[1] += score_cand_b
      step.append(score_cand_b)
      x_now[2:4] = x_cand_b
    else:
      score[1] += score_stay_b
      step.append(score_stay_b)
    
    d_now = (x_now[0]-x_now[2])**2 + (x_now[1]-x_now[3])**2
    x_list = np.row_stack((x_list, x_now))
    score_ts = np.row_stack((score_ts,score))
    step_ts = np.row_stack((step_ts,step))
    
    # The score (total log population) equals the sum of resources at each time point
  
  if not quiet:
    if type(ax) == type(None): _,ax = plt.subplots(1,3,figsize = (22,7))
    X,Y = np.meshgrid(np.linspace(-6,6,1024), np.linspace(-6,6,1024))
    z = f(X,Y)
    ax[0].contourf(X,Y,z, levels = 20, cmap = 'hot')
    l = mpl.lines.Line2D(x_list[:,0], x_list[:,1],c = 'b')
    ax[0].add_line(l)
    m = mpl.lines.Line2D(x_list[:,2], x_list[:,3],c = 'g')
    ax[0].add_line(m)
    ax[0].set_xlim([-6,6]); ax[0].set_ylim([-6,6])
    ax[1].plot(np.arange(0,max_iter),score_ts[:,0],'b',label = f'alpha = {alpha[0]:.3f}')
    ax[1].plot(np.arange(0,max_iter),score_ts[:,1],'g',label = f'alpha = {alpha[1]:.3f}')
    ax[1].set_ylabel('Cumulative score')
    ax[1].legend()
    ax[2].plot(np.arange(0,max_iter),step_ts[:,0],'b',label = f'alpha = {alpha[0]:.3f}')
    ax[2].plot(np.arange(0,max_iter),step_ts[:,1],'g',label = f'alpha = {alpha[1]:.3f}')
    ax[2].set_ylabel('Single-step score')
    ax[2].legend()
    plt.title(f'beta = {beta}')
    plt.show()
    print(f'Total score A = {score[0]:.3f} (alpha = {alpha[0]:.3f})')
    print(f'Total score B = {score[1]:.3f} (alpha = {alpha[1]:.3f})')
    
  return score, score_ts, step_ts, x_list

for _ in range(4):
  f = generate_f()
  x_anchor = rng.uniform(-4,4,2).repeat(2)[[0,2,1,3]]
  for beta in [0.04,0.2,1,5,25]:
    sim_competition([20,200],f,beta = beta, x_init = x_anchor,max_iter = 1500,
                    quiet = False)

def sim_competition_l(alpha, # size 2
                   f = generate_f(),
                   beta = 0.3, # competition parameter
                   x_init = np.random.uniform(-4,4,size=2), # initial coordinates
                   max_iter = 1000, movement_rate = 0.05, quiet = True, ax = None):
  if alpha[0] < 0 or alpha[1] < 0: raise ValueError
  if len(x_init) == 2: x_init = x_init.repeat(2)[[0,2,1,3]]
  x_list = x_init.copy() # list of coordinates across 100 iterations
  x_now = x_init.copy()
  d_now = (x_init[0]-x_init[2])**2 + (x_init[1]-x_init[3])**2
  d_now = d_now ** 0.5
  score = np.array([f(x_now[0], x_now[1]), f(x_now[2],x_now[3])]) # 2 scores
  score_now = score
  score_now[0] *= sts.norm.pdf(0, scale = beta) / \
    (sts.norm.pdf(0, scale = beta) + \
     (score[1]/score[0]) * sts.norm.pdf(d_now, scale = beta))
  score_now[1] *= sts.norm.pdf(0, scale = beta) / \
    (sts.norm.pdf(0, scale = beta) + \
     (score[0]/score[1]) * sts.norm.pdf(d_now, scale = beta))
  # In the competition, we assume the population of each strain of bacteria to 
  # be exponentially growing with the score per step, and they spread normally 
  # by sd = beta,  
  # and accordingly occupy the resources of the other strain, thus multiplying
  # the score above.
  # We use log population as the score
  
  score = score_now # we re-set the scores
  score_ts = score
  step_ts = score
  
  # we make the two items to take turns in determining the new location
  for _ in range(max_iter - 1):
    x_cand_a = rng.multivariate_normal(mean = x_now[0:2],cov = np.diag([movement_rate]*2))
    d_cand_a = (x_cand_a[0]-x_now[2])**2 + (x_cand_a[1]-x_now[3])**2
    score_cand_a = f(x_cand_a[0],x_cand_a[1])
    score_cand_a *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       (score[1]/score[0]) * sts.norm.pdf(d_cand_a, scale = beta))
    score_stay_a = f(x_now[0],x_now[1])
    score_stay_a *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       (score[1]/score[0]) * sts.norm.pdf(d_now, scale = beta))
    prob_a = sp.special.expit((score_cand_a-score_stay_a)*alpha[0])
    # when evaluating the occupation of resources by B, we assume B to stay
    # this is a heuristic to assist computation
    # since we need to apply the same heuristic to B (to be fair)
    # we update the values later
        
    x_cand_b = rng.multivariate_normal(mean = x_now[2:4],cov = np.diag([movement_rate]*2))
    d_cand_b = (x_cand_b[0]-x_now[0])**2 + (x_cand_b[1]-x_now[1])**2
    score_cand_b = f(x_cand_b[0],x_cand_b[1])
    score_cand_b *=  sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       (score[0]/score[1]) * sts.norm.pdf(d_cand_b, scale = beta))
    score_stay_b = f(x_now[2],x_now[3])
    score_stay_b *= sts.norm.pdf(0, scale = beta) / \
      (sts.norm.pdf(0, scale = beta) + \
       (score[0]/score[1]) * sts.norm.pdf(d_now, scale = beta))
    prob_b = sp.special.expit((score_cand_b-score_stay_b)*alpha[1])
    
    step = []
    if rng.random() < prob_a: # move to the next location (a/b independently)
      score[0] += score_cand_a
      step.append(score_cand_a)
      x_now[0:2] = x_cand_a
    else: # we do not update x_now
      score[0] += score_stay_a
      step.append(score_stay_a)
    
    if rng.random() < prob_b:
      score[1] += score_cand_b
      step.append(score_cand_b)
      x_now[2:4] = x_cand_b
    else:
      score[1] += score_stay_b
      step.append(score_stay_b)
    
    d_now = (x_now[0]-x_now[2])**2 + (x_now[1]-x_now[3])**2
    x_list = np.row_stack((x_list, x_now))
    score_ts = np.row_stack((score_ts,score))
    step_ts = np.row_stack((step_ts,step))
    
    # The score (total log population) equals the sum of resources at each time point
  
  if not quiet:
    if type(ax) == type(None): _,ax = plt.subplots(1,3,figsize = (22,7))
    X,Y = np.meshgrid(np.linspace(-6,6,1024), np.linspace(-6,6,1024))
    z = f(X,Y)
    ax[0].contourf(X,Y,z, levels = 20, cmap = 'hot')
    l = mpl.lines.Line2D(x_list[:,0], x_list[:,1],c = 'b')
    ax[0].add_line(l)
    m = mpl.lines.Line2D(x_list[:,2], x_list[:,3],c = 'g')
    ax[0].add_line(m)
    ax[0].set_xlim([-6,6]); ax[0].set_ylim([-6,6])
    ax[1].plot(np.arange(0,max_iter),score_ts[:,0],'b',label = f'alpha = {alpha[0]:.3f}')
    ax[1].plot(np.arange(0,max_iter),score_ts[:,1],'g',label = f'alpha = {alpha[1]:.3f}')
    ax[1].set_ylabel('Cumulative score')
    ax[1].legend()
    ax[2].plot(np.arange(0,max_iter),step_ts[:,0],'b',label = f'alpha = {alpha[0]:.3f}')
    ax[2].plot(np.arange(0,max_iter),step_ts[:,1],'g',label = f'alpha = {alpha[1]:.3f}')
    ax[2].set_ylabel('Single-step score')
    ax[2].legend()
    plt.title(f'beta = {beta}')
    plt.show()
    print(f'Total score A = {score[0]:.3f} (alpha = {alpha[0]:.3f})')
    print(f'Total score B = {score[1]:.3f} (alpha = {alpha[1]:.3f})')
    
  return score, score_ts, step_ts, x_list

for _ in range(4):
  f = generate_f()
  x_anchor = rng.uniform(-4,4,2).repeat(2)[[0,2,1,3]]
  for beta in [0.04,0.2,1,5,25]:
    sim_competition_l([20,200],f,beta = beta, x_init = x_anchor,max_iter = 1500,
                    quiet = False)