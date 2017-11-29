import numpy as np
from util.util import softmax_prob, Message, discount, fmt_row
from util.frozen_lake import rollout
import pdb

def value_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS,dtype='int')]  
  for it in range(nIt):
    V, pi = vstar_backup(Vs[-1], env, gamma)
    Vs.append(V)
    pis.append(pi)
  return Vs, pis

def policy_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS,dtype='int')] 
  for it in range(nIt):
    vpi = policy_evaluation_v(pis[-1], env, gamma)
    qpi = policy_evaluation_q(vpi, env, gamma)
    pi = qpi.argmax(axis=1)
    Vs.append(vpi)
    pis.append(pi)
  return Vs, pis


def policy_gradient_optimize(env, policy, gamma,
      max_pathlength, timesteps_per_batch, n_iter, stepsize):
  from collections import defaultdict
  stat2timeseries = defaultdict(list)
  widths = (17,10,10,10,10)
  print fmt_row(widths, ["EpRewMean","EpLenMean","Perplexity","KLOldNew"])
  for i in xrange(n_iter):
      # collect rollouts
      total_ts = 0
      paths = [] 
      while True:
          path = rollout(env, policy, max_pathlength)                
          paths.append(path)
          total_ts += path["rewards"].shape[0] # Number of timesteps in the path
          #pathlength(path)
          if total_ts > timesteps_per_batch: 
              break

      # get observations:
      obs_no = np.concatenate([path["observations"] for path in paths])
      # Update policy
      policy_gradient_step(env, policy, paths, gamma, stepsize)

      # Compute performance statistics
      pdists = np.concatenate([path["pdists"] for path in paths])
      kl = policy.compute_kl(pdists, policy.compute_pdists(obs_no)).mean()
      perplexity = np.exp(policy.compute_entropy(pdists).mean())

      stats = {  "EpRewMean" : np.mean([path["rewards"].sum() for path in paths]),
                 "EpRewSEM" : np.std([path["rewards"].sum() for path in paths])/np.sqrt(len(paths)),
                 "EpLenMean" : np.mean([path["rewards"].shape[0] for path in paths]), #pathlength(path) 
                 "Perplexity" : perplexity,
                 "KLOldNew" : kl }
      print fmt_row(widths, ['%.3f+-%.3f'%(stats["EpRewMean"], stats['EpRewSEM']), stats['EpLenMean'], stats['Perplexity'], stats['KLOldNew']])
      
      for (name,val) in stats.items():
          stat2timeseries[name].append(val)
  return stat2timeseries



#####################################################
## TODO: You need to implement all functions below ##
#####################################################
def vstar_backup(v_n, env, gamma):
  """
  Apply Bellman backup operator V -> T[V], i.e., perform one step of value iteration

  :param v_n: the state-value function (1D array) for the previous iteration
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: a pair (v_p, a_p), where 
  :  v_p is the updated state-value function and should be a 1D array (S -> R),
  :  a_p is the updated (deterministic) policy, which should also be a 1D array (S -> A)
  """
  #YOUR_CODE_HERE # TODO
  q = np.zeros([env.nS,env.nA])
  v_p = np.zeros(np.shape(v_n))
  a_p = np.zeros(np.shape(v_n))
  for i in range(env.nS):
      for j in range(env.nA):
          a = np.array(env.P[i][j])
          a = np.vstack(a)
          q[i,j] = np.sum(np.multiply(a[:,0].astype(np.float), a[:,2].astype(np.float))) + gamma*(np.sum(np.multiply(a[:,0].astype(np.float), v_n[list(a[:,1].astype(np.int32))])))
      v_p[i] = np.max(q[i,:])
      a_p[i] = np.argmax(q[i,:])
  assert v_p.shape == (env.nS,)
  assert a_p.shape == (env.nS,)  
  return (v_p, a_p)

def policy_evaluation_v(pi, env, gamma):
  """
  :param pi: a deterministic policy (1D array: S -> A)
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: vpi, the state-value function for the policy pi
  
  Hint: use np.linalg.solve
  """
  #YOUR_CODE_HERE # TODO
  pi = pi.astype(np.int32)
  vpi = np.zeros(np.shape(pi))
  I = np.eye(env.nS)
  reward = np.zeros(np.shape(pi))
  p = np.zeros([env.nS, env.nS])
  for i in range(env.nS):
      for j in (env.P[i][pi[i]]):
          reward[i] += j[0]*j[2]
          p[i][j[1]] = j[0]
  
  vpi = np.linalg.solve((I-gamma*p), reward)
  assert vpi.shape == (env.nS,)
  return vpi

def policy_evaluation_q(vpi, env, gamma):
  """
  :param vpi: the state-value function for the policy pi
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: qpi, the state-action-value function for the policy pi
  """
  qpi = np.zeros([env.nS,env.nA])
  for i in range(env.nS):
      for j in range(env.nA):
          a = env.P[i][j]
          qpi_temp = 0
          reward = 0
          for k in range(np.shape(a)[0]):
              reward += a[k][0] *a[k][2]
              qpi_temp += a[k][0] * vpi[a[k][1]]
          qpi_temp *= gamma
          qpi_temp += reward
          qpi[i][j] = qpi_temp
          
  assert qpi.shape == (env.nS, env.nA)
  return qpi

def softmax_policy_gradient(f_sa, s_n, a_n, adv_n):
  """
  Compute policy gradient of policy for discrete MDP, where probabilities
  are obtained by exponentiating f_sa and normalizing.
  
  See softmax_prob and softmax_policy_checkfunc functions in util. This function
  should compute the gradient of softmax_policy_checkfunc.
  
  INPUT:
    f_sa : a matrix representing the policy parameters, whose first dimension s 
           indexes over states, and whose second dimension a indexes over actions
    s_n : states (vector of int)
    a_n : actions (vector of int)
    adv_n : discounted long-term returns (vector of float)
  """
  print "Entered softmax"
  A = softmax_prob(f_sa)
  gradient = np.zeros(f_sa.shape)
  for timestep in range(np.shape(s_n)[0]):
      for current_action in range(np.shape(f_sa)[1]):
          if current_action == a_n[timestep]:
              reward = (1 - A[s_n[timestep], a_n[timestep]])*adv_n[timestep]
              gradient[s_n[timestep], current_action] += reward
          else:
              reward = -(A[s_n[timestep], current_action])*adv_n[timestep]
              gradient[s_n[timestep], current_action] += reward      
          #gradient[s_n[timestep], a_n[timestep]] += reward
  grad_sa = gradient/np.shape(s_n)[0]
  #assert grad_sa == (env.nS, env.nA)
  return grad_sa


def policy_gradient_step(env, policy, paths, gamma,stepsize):
  """
  Compute the discounted returns, compute the policy gradient (using softmax_policy_gradient above),
  and update the policy parameters policy.f_sa
  """
  theta = policy.f_sa
  grad = np.zeros([env.nS, env.nA])
  num_paths = np.shape(paths)[0]
  for i in range(num_paths):
      x = paths[i]['observations']
      u = paths[i]['actions']
      g_temp = np.zeros(np.shape(x))
      g = np.zeros(np.shape(x))
      for j in range(len(x)):
          g_temp[j] = gamma**j*paths[i]['rewards'][j]
      for j in range(len(x)):
          g[j] = np.sum(g_temp[j:len(g)])
      grad += softmax_policy_gradient(theta, x, u, g)
  
    
  grad = grad/num_paths
  policy.f_sa += stepsize*grad
  #YOUR_CODE_HERE # TODO

