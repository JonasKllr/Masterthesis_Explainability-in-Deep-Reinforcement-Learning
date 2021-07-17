import math
import pickle
import random

import numpy as np
import torch
from scipy.stats import rankdata


class UniformReplayBuffer:
    def __init__(self, action_dim, state_dim, n_steps, gamma, buffer_length, batch_size, device):
        """A simple replay buffer with uniform sampling."""

        self.max_size   = buffer_length
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device
        
        self.s  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)
    
    def add(self, s, a, r, s2, d):
        """s and s2 are np.arrays of shape (state_dim,)."""
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:  torch.Size([batch_size, state_dim])
        a:  torch.Size([batch_size, action_dim])
        r:  torch.Size([batch_size, 1])
        s2: torch.Size([batch_size, state_dim])
        d:  torch.Size([batch_size, 1])"""

        if self.n_steps == 1:
            # sample index
            ind = np.random.randint(low = 0, high = self.size, size = self.batch_size)

            return (torch.tensor(self.s[ind]).to(self.device), 
                    torch.tensor(self.a[ind]).to(self.device), 
                    torch.tensor(self.r[ind]).to(self.device), 
                    torch.tensor(self.s2[ind]).to(self.device), 
                    torch.tensor(self.d[ind]).to(self.device))
        else:
            # sample index
            ind = np.random.randint(low = 0, high = self.size - (self.n_steps - 1), size = self.batch_size)

            # get s, a
            s = self.s[ind]
            a = self.a[ind]

            # get s', d
            s_n = self.s2[ind + (self.n_steps - 1)]
            d   = self.d[ind + (self.n_steps - 1)]

            # compute reward part of n-step return
            r_n = np.zeros((self.batch_size, 1), dtype=np.float32)

            for i, idx in enumerate(ind):
                for j in range(self.n_steps):
                    
                    # add discounted reward
                    r_n[i] += (self.gamma ** j) * self.r[idx + j]
                    
                    # if done appears, break and set done which will be returned True (to avoid incorrect Q addition)
                    if self.d[idx + j] == 1:
                        d[i] = 1
                        break

            return (torch.tensor(s).to(self.device), 
                    torch.tensor(a).to(self.device), 
                    torch.tensor(r_n).to(self.device), 
                    torch.tensor(s_n).to(self.device), 
                    torch.tensor(d).to(self.device))

class Proportional_PER_Buffer:
    def __init__(self, action_dim, state_dim, n_steps, gamma, buffer_length, batch_size, alpha, beta_start, beta_inc, device):
        """PER buffer using the proportional approach.
        Constructs a SumTree (and MinTree) to allow for efficient O(log n) prioritized sampling.
        
        Note regarding naming conventions: 
        'prio' refers to |TD_error|. Consequently, max_prio describes max |TD_error|.
        'prio_a' refers to (|TD_error| + epsilon)^{alpha} or (|TD_error|)^{alpha}, respectively.
        epsilon is considered in the 'update_prio' method, but not in the 'add' method for new samples, 
        as the latter prio can by construction not be close to 0 anyways.
        """

        assert n_steps == 1, "Currently, PER is only supported with n = 1."

        # set buffer size to lowest power of 2 above provided buffer length
        max_size = 1
        while max_size < buffer_length:
            max_size *= 2

        self.max_size   = max_size
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device

        # PER related params
        self.max_prio = 1.0
        self.alpha    = alpha
        self.beta     = beta_start
        self.beta_inc = beta_inc

        # construct sum tree and min tree for prio_as (each having max_size-1 parent nodes and max_size leafs)
        self.sumtree = np.zeros(2*self.max_size-1)
        self.mintree = np.ones(2*self.max_size-1) * np.inf

        # construct buffer        
        self.s  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)

    def _buff_to_tree_idx(self, buff_idx):
        """Converts the buffer idx in the tree idx to access the Min- and SumTree."""
        return buff_idx + self.max_size - 1
   
    def _tree_to_buff_idx(self, tree_idx):
        """Converts the tree idx back to the buffer idx."""
        return tree_idx - self.max_size + 1

    def add(self, s, a, r, s2, d):
        """Adds transition to buffer. Priority is set to the maximum priority ever seen.
        s and s2 are np.arrays of shape (state_dim,)."""
        
        # update trees by assigning max prio to new samples
        tree_idx = self._buff_to_tree_idx(self.ptr)

        self._update_single_sumtree_idx(tree_idx=tree_idx, prio_a=self.max_prio ** self.alpha)
        self._update_single_mintree_idx(tree_idx=tree_idx, prio_a=self.max_prio ** self.alpha)

        # update buffer
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        # increase ptr and size
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _update_single_sumtree_idx(self, tree_idx, prio_a):
        """Updates SumTree for a new prio_a of a given leaf node.

        tree_idx: int
        prio_a:   float
        """
        change = prio_a - self.sumtree[tree_idx]
        self.sumtree[tree_idx] = prio_a

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.sumtree[tree_idx] += change

    def _update_single_mintree_idx(self, tree_idx, prio_a):
        """Updates MinTree for a new prio_a of a given leaf node.

        tree_idx: int
        prio_a:   float
        """
        # case 1: new value is smaller or equal than old one, run through tree and set new mins
        if prio_a <= self.mintree[tree_idx]:
            
            # replace old by new value
            self.mintree[tree_idx] = prio_a

            while tree_idx != 0:
                
                # jump to next tree level
                tree_idx = (tree_idx - 1) // 2

                # replace parent if parent is larger than new value, else break
                if prio_a < self.mintree[tree_idx]:
                    self.mintree[tree_idx] = prio_a
                else:
                    break
        
        # case 2: new value is larger than old one
        else:

            # replace old by new value
            self.mintree[tree_idx] = prio_a

            while tree_idx != 0:

                # get value of neighbour
                if tree_idx % 2 != 0:
                    neighbour = self.mintree[tree_idx + 1]
                else:
                    neighbour = self.mintree[tree_idx - 1]

                # go up one tree level
                tree_idx = (tree_idx - 1) // 2

                # If parent had neighbour's value beforehand, we can stop.                 
                # Else we got to replace parent by minimum of neighbour and prio_a.
                if self.mintree[tree_idx] == neighbour:
                    break
                else:
                    prio_a = min(prio_a, neighbour)
                    self.mintree[tree_idx] = prio_a
        
    def sample(self):
        """Return sizes:
        s:          torch.Size([batch_size, state_dim])
        a:          torch.Size([batch_size, action_dim])
        r:          torch.Size([batch_size, 1])
        s2:         torch.Size([batch_size, state_dim])
        d:          torch.Size([batch_size, 1])
        weights:    torch.Size([batch_size, 1])
        tree_idx:   list of length batch_size
        """
        # increase beta
        self.beta = min(1., self.beta + self.beta_inc)

        # get sampling segments
        segment = self.sumtree[0] / self.batch_size

        # sample tree indices according to their prio_as
        tree_idxs = []
        prio_as = []

        for i in range(self.batch_size):
            rnd = random.random() * segment + i * segment

            tree_idx, prio_a = self._prefixsum_tree_idx(rnd)
            tree_idxs.append(tree_idx)
            prio_as.append(prio_a)
        
        buff_idxs = [0, 3, 5]
        tree_idx = list(self._buff_to_tree_idx(np.array(buff_idxs)))
        prio_as = list(self.sumtree[tree_idx])

        # get corresponding buffer indices
        buff_idxs = self._tree_to_buff_idx(np.array(tree_idxs))

        # calculate importance sampling weights from sampled prio_as
        prio_as = np.array(prio_as)
        probs = prio_as / self.sumtree[0]
        weights = (probs * self.size) ** (-self.beta)

        # normalize weights by max_weight

        # Note: Max weight relates to the maximum weight of ALL stored transitions, not just the sampled ones.
        #       This follows the OpenAI baseline implementation. However, using the max of the sample is possible as well.
        #       In the latter case, there would be no need to maintain the MinTree.

        max_weight = ((self.mintree[0] / self.sumtree[0]) * self.size) ** (-self.beta)
        weights /= max_weight

        # give weights shape (batch_size, 1)
        weights = weights.reshape(self.batch_size, 1)

        # return
        return (torch.tensor(self.s[buff_idxs]).to(self.device), 
                torch.tensor(self.a[buff_idxs]).to(self.device), 
                torch.tensor(self.r[buff_idxs]).to(self.device), 
                torch.tensor(self.s2[buff_idxs]).to(self.device), 
                torch.tensor(self.d[buff_idxs]).to(self.device),
                torch.tensor(weights).to(self.device),
                tree_idxs)

    def _prefixsum_tree_idx(self, rnd):
        """Returns for a given float (rnd) in [0, priority_sum] the corresponding (sum) tree index and its prio_a."""
        parent_index = 0

        while True:

            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # End search if bottom of tree is reached
            if left_child_index >= len(self.sumtree):
                leaf_index = parent_index
                break
            else:
                if rnd <= self.sumtree[left_child_index]:
                    parent_index = left_child_index
                else:
                    rnd -= self.sumtree[left_child_index]
                    parent_index = right_child_index

        # return (tree) index and prio_a of found leaf
        return leaf_index, self.sumtree[leaf_index]

    def update_prio(self, idx, TD_error):
        """Updates SumTree for given tree indices and TD errors.

        idx: list of length batch_size
        TD_error: torch.Size([batch_size, 1])
        """

        prios = TD_error.detach().cpu().numpy().reshape(self.batch_size)

        for i, tree_i in enumerate(idx):
            prio = prios[i]

            self._update_single_sumtree_idx(tree_idx = tree_i, prio_a = (prio + 1e-5) ** self.alpha)
            self._update_single_mintree_idx(tree_idx = tree_i, prio_a = (prio + 1e-5) ** self.alpha)

            self.max_prio = max(self.max_prio, prio)
            

    def print_trees(self):
        """Prints SumTree and MinTree. Only useful for small trees."""
        tree_level = math.ceil(math.log(self.max_size + 1, 2))

        print("------- SUM TREE --------")
        for k in range(1, tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.sumtree[j], end=' ')
            print()

        print("------- MIN TREE --------")
        for k in range(1, tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.mintree[j], end=' ')
            print()

class Inefficient_Proportional_PER_Buffer:
    def __init__(self, action_dim, state_dim, n_steps, gamma, buffer_length, batch_size, alpha, beta_start, beta_inc, device):
        """PER buffer using the proportional approach.
        This is an inefficient implementation for debugging purposes. Complexity of prioritized sampling is O(n)."""

        assert n_steps == 1, "Currently, PER is only supported with n = 1."

        # set buffer size to lowest power of 2 above provided buffer length (kept in this class for comparison purposes)
        max_size = 1
        while max_size < buffer_length:
            max_size *= 2

        self.max_size   = max_size
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device

        # PER related params
        self.max_prio = 1.0
        self.alpha    = alpha
        self.beta     = beta_start
        self.beta_inc = beta_inc

        # construct prio_as array (stores prio^{alpha}, where prio = |TD_error| + epsilon)
        self.prio_as = np.zeros(self.max_size)

        # construct buffer        
        self.s  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        """ Adds transition to buffer. Priority is set to the maximum priority ever seen.

        s and s2 are np.arrays of shape (state_dim,)."""
        
        # update prio_as using maximum priority seen so far
        self.prio_as[self.ptr] = self.max_prio ** self.alpha

        # update buffer
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        # increase ptr and size
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        """Return sizes:
        s:       torch.Size([batch_size, state_dim])
        a:       torch.Size([batch_size, action_dim])
        r:       torch.Size([batch_size, 1])
        s2:      torch.Size([batch_size, state_dim])
        d:       torch.Size([batch_size, 1])
        weights: torch.Size([batch_size, 1])
        idx:     list of length batch_size
        """
        # increase beta
        self.beta = min(1., self.beta + self.beta_inc)

        # calculate probabilities 
        # Note: One could also divide by np.sum(self.prio_as) in the following, as the remaining entries are 0 anyways. 
        #       This is what happens in the SumTree implementation.
        prio_as = self.prio_as[:self.size]
        probs = prio_as / np.sum(prio_as)

        # sample transition indices
        idx = np.random.choice(self.size, self.batch_size, p = probs)

        # calculate importance sampling weights from sampled prio_as
        sample_probs = probs[idx]
        weights = (sample_probs * self.size) ** (-self.beta)

        # normalize weights by max_weight (again: maximum weight over ALL transitions in the buffer)
        max_weight = (np.min(probs) * self.size) ** (-self.beta)
        weights /= max_weight

        # give weights shape (batch_size, 1)
        weights = weights.reshape(self.batch_size, 1)

        # return
        return (torch.tensor(self.s[idx]).to(self.device), 
                torch.tensor(self.a[idx]).to(self.device), 
                torch.tensor(self.r[idx]).to(self.device), 
                torch.tensor(self.s2[idx]).to(self.device), 
                torch.tensor(self.d[idx]).to(self.device),
                torch.tensor(weights).to(self.device),
                idx)

    def update_prio(self, idx, TD_error):
        """Updates prio_as array for given indices and TD errors.

        idx:      list of length batch_size
        TD_error: torch.Size([batch_size, 1])
        """

        prios = TD_error.detach().cpu().numpy().reshape(self.batch_size)

        for i, buff_idx in enumerate(idx):
            prio = prios[i]

            # update prio_as
            self.prio_as[buff_idx] = (prio + 1e-5) ** self.alpha
            
            # update maximum priority ever seen
            self.max_prio = max(self.max_prio, prio)

class Inefficient_Rank_PER_Buffer:
    def __init__(self, action_dim, state_dim, n_steps, gamma, buffer_length, batch_size, alpha, beta_start, beta_inc, device):
        """PER buffer using the rank-based approach.
        For now, this is an inefficient implementation without usage of the more appropriate binary heap data structure."""

        assert n_steps == 1, "Currently, PER is only supported with n = 1."

        self.max_size   = buffer_length
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device

        # PER related params
        self.max_TD_error = 1.0
        self.alpha        = alpha
        self.beta         = beta_start
        self.beta_inc     = beta_inc

        # construct TD_error array
        self.TD_errors = np.zeros(self.max_size)

        # construct buffer        
        self.s  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        """ Adds transition to buffer. 
        TD_error is set to the maximum TD_error ever seen, so that the transition's priority becomes maximal.

        s and s2 are np.arrays of shape (state_dim,)."""
        
        # update TD_error using maximum seen so far
        self.TD_errors[self.ptr] = self.max_TD_error

        # update buffer
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d

        # increase ptr and size
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        """Return sizes:
        s:       torch.Size([batch_size, state_dim])
        a:       torch.Size([batch_size, action_dim])
        r:       torch.Size([batch_size, 1])
        s2:      torch.Size([batch_size, state_dim])
        d:       torch.Size([batch_size, 1])
        weights: torch.Size([batch_size, 1])
        idx:     list of length batch_size
        """
        # increase beta
        self.beta = min(1., self.beta + self.beta_inc)

        # 1. calculate ranks
        ranks = rankdata(self.TD_errors[:self.size])

        # 2. calculate prio_as
        prio_as = (1 / ranks) ** self.alpha

        # 3. calculate probabilities
        probs = prio_as / np.sum(prio_as)

        # sample transition indices
        idx = np.random.choice(self.size, self.batch_size, p = probs)

        # calculate importance sampling weights from sampled prio_as
        sample_probs = probs[idx]
        weights = (sample_probs * self.size) ** (-self.beta)

        # normalize weights by max_weight
        
        # Note: Here we use the sample-based maximum. 
        #       This follows the code implementation of the paper 'Improving DDPG via Prioritized Experience Replay'.
        
        weights /= np.max(weights)

        # give weights shape (batch_size, 1)
        weights = weights.reshape(self.batch_size, 1)

        # return
        return (torch.tensor(self.s[idx]).to(self.device), 
                torch.tensor(self.a[idx]).to(self.device), 
                torch.tensor(self.r[idx]).to(self.device), 
                torch.tensor(self.s2[idx]).to(self.device), 
                torch.tensor(self.d[idx]).to(self.device),
                torch.tensor(weights).to(self.device),
                idx)

    def update_prio(self, idx, TD_error):
        """Updates TD_error array for given indices and TD errors.

        idx: list of length batch_size
        TD_error: torch.Size([batch_size, 1])
        """

        TD_error = TD_error.detach().cpu().numpy().reshape(self.batch_size)

        for i, buff_idx in enumerate(idx):
            TD_err = TD_error[i]

            # update TD_error
            self.TD_errors[buff_idx] = TD_err
            
            # update maximum TD_error ever seen
            self.max_TD_error = max(self.max_TD_error, TD_err)

"""
buffer_length = 8
batch_size = 8
alpha = 1.

ineff_buff = Inefficient_Proportional_PER_Buffer(1, 1, 1, 1, buffer_length, batch_size, alpha, 1, 0, "cpu")
eff_buff = Proportional_PER_Buffer(1, 1, 1, 1, buffer_length, batch_size, alpha, 1, 0, "cpu")

ineff_buff.prio_as = np.array([2, 3, 1, 0.5, 4, 2.3, 7, 5])
eff_buff.update_prio(eff_buff._buff_to_tree_idx(np.array([0, 1, 2, 3, 4, 5, 6, 7])), torch.tensor(np.array([2, 3, 1, 0.5, 4, 2.3, 7, 5]) - 1e-5))
eff_buff.print_trees()

eff_buff.batch_size = 1
eff_buff.update_prio(eff_buff._buff_to_tree_idx(np.array([5])), torch.tensor(np.array([4.2]) - 1e-5))
eff_buff.print_trees()
"""
"""
for i in range(10):

    np.random.seed(i)
    if i < 5:
        ineff_buff.max_prio = i
        eff_buff.max_prio = i
    else:
        ineff_buff.max_prio = i + 1
        eff_buff.max_prio = i + 1

    s = np.random.randn() + 3
    a = np.random.randn()
    r = np.random.randn()
    s2 = np.random.randn()
    d = True

    ineff_buff.add(s, a, r, s2, d)
    eff_buff.add(s, a, r, s2, d)

print(ineff_buff.prio_as)
eff_buff.print_trees()

print(eff_buff.s)
print(eff_buff.sample())
"""
"""
print("------------------ SAMPLE -----------------------")
np.random.seed(10237)
cnt = np.zeros(8)

for i in range(1, 1000000):
    *rest, idx = eff_buff.sample()
    cnt[eff_buff._tree_to_buff_idx(np.array(idx))] += 1

    if i % 50000 == 0:
        print(cnt / i)

print(cnt/ 1000000)

#*batch, weight, idx = ineff_buff.sample()
#print(weight, idx)

#*batch, weight, idx = eff_buff.sample()
#print(weight, idx)
"""