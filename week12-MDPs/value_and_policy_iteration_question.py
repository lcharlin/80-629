#!/usr/bin/env python
# coding: utf-8

# # Week 12 - Sequential Decision Making I
# ## Value and Policy Iteration Exercices

# Author: Massimo Caccia massimo.p.caccia@gmail.com <br>
# 
# The code was Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl <br>
# and then from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo

# ## 0. Preliminaries
# 
# Before we jump into the value and policy iteration excercies, we will test your comprehension of a Markov Decision Process (MDP). <br>

# ### 0.1 Tic-Tac-Toe
# 
# Let's take a simple example: Tic-Tac-Toe (also known as Tic-tac-toe, noughts and crosses, or Xs and Os). Definition: it is a paper-and-pencil game for two players, X and O, who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.

# In[1]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://bjc.edc.org/bjc-r/img/3-lists/TTT1_img/Three%20States%20of%20TTT.png")


# **Question:** Imagine you were trying to build an agent for this game. Let's try to describe how we would model it. Specifically, what are the states, actions, transition function and rewards?

# States: 
# 
# Actions: 
# 
# Reward: 
# 
# Transition Probabilities: 

# ### 0.2 Recommender Systems
# 
# **Question:** In the last class we discussed recommender systems. Imagine that you would like to model the recommendation process over time as an MDP. How would you do it?

# States: 
# 
# Actions: 
# 
# Reward: 
# 
# Transition Probabilities: 

# ## 1. Value Iteration

# The exercises will test your capacity to **complete the value iteration algorithm**.
# 
# You can find details about the algorithm at slide 46 of the [slide](http://www.cs.toronto.edu/~lcharlin/courses/80-629/slides_rl.pdf) deck. <br>
# 
# The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12.

# ### 1.1 Setup

# In[18]:


#imports
get_ipython().system('wget -nc https://raw.githubusercontent.com/lcharlin/80-629/master/week12-MDPs/gridWorldGame.py')
import numpy as np
from gridWorldGame import standard_grid, negative_grid, print_values, print_policy


# Let's set some variables. <br>
# `SMALL_ENOUGH` is a threshold we will utilize to determine the convergence of value iteration<br>
# `GAMMA` is the discount factor denoted $\gamma$ in the slides (see slide 36) <br>
# `ALL_POSSIBLE_ACTIONS` are the actions you can take in the GridWold, as in slide 12. In this simple grid world, we will have four actions: Up, Down, Right, Left. <br>
# `NOISE_PROB` defines how stochastic the environement is. It is the probability that the environment takes you where a random action would. 

# In[3]:


SMALL_ENOUGH = 1e-3 # threshold to declare convergence
GAMMA = 0.9         # discount factor 
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R') # Up, Down, Left, Right
NOISE_PROB = 0.1    # Probability of the agent not reaching it's intended goal after an action


# Now we will set up a the Gridworld. <br>
# It has the following Rewards:

# In[4]:


grid = standard_grid(noise_prob=NOISE_PROB)
print("rewards:")
print_values(grid.rewards, grid)


# There are three absorbing states: (0,3),(1,3), and (1,1)

# Next, we will define a random inital policy $\pi$. <br>
# Remember that a policy maps states to actions $\pi : S \rightarrow A$.

# In[5]:


policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)


# The N/A correspond to absorbing states.

# Next, we will randomly initialize the value fonction

# In[16]:


np.random.seed(1234) # make sure this is reproducable

V = {}
states = grid.all_states()
for s in states:
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# initial value for all states in grid
print_values(V, grid)


# Note that we set to Null the values of the terminal states. <br> 
# For the print_values() function to compile, we set them to 0.

# ### 1.2 Value iteration algorithms - code completion
# 
# You will now have to complete the Value iteration algorithm. <br>
# Remember that, for each iteration, the value of each state s is updated using:
# 
# $$
# V(s) = \underset{a}{max}\big\{ \sum_{s',a}  p(s'|s,a)(r + \gamma*V(s') \big\}
# $$
# Note that in the current gridWorld, $p(s'|s,a)$ is deterministic. <br>
# Also, remember that in value iteration, the policy is implicit. <br> Thus, you don't need to update it at every iteration. <br>
# Run the algorithm until convergence.
# 

# In[15]:


iteration=1
while True: # run the algorithm until convergence
    print("iteration %d: " % iteration)
    print_values(V, grid)
    print("\n\n")

    biggest_change = 0
  
    for s in states: # for each state
        old_v = V[s]

        # V(s) only has a value if it's not a terminal/absorbing state
        if s in policy:
        
              new_v = float('-inf')
              for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                # get reward
                r = grid.move(a)
                # get s'
                sprime = grid.current_state()

                ## Implement This!
                ## hints:
                ##   - compute this V[s] = max[a]{ sum[s'] { p(s'|s,a)[r + gamma*V[s']] } }

                if v > new_v:
                    new_v = v
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

                
        if biggest_change < SMALL_ENOUGH:
            break

    iteration += 1


# Now that the value function is trained, use it to find the optimal policy.

# In[8]:


deterministic_grid = standard_grid(noise_prob=0.)

for s in policy.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in ALL_POSSIBLE_ACTIONS:
        deterministic_grid.set_state(s)
        r = deterministic_grid.move(a)
        v = r + GAMMA * V[deterministic_grid.current_state()]
        if v > best_value:
            best_value = v
            best_a = a
    policy[s] = best_a


# Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.

# In[9]:


print("values:")
print_values(V, grid)
print("policy:")
print_policy(policy, grid)


# ## 2. Policy Iteration

# You will be tested on your capacity to **complete the poliy iteration algorithm**. <br>
# You can find details about the algorithm at slide 47 of the slide deck. <br>
# The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12. <br>
# This Gridworld is however simpler because the MDP is deterministic. <br>

# First we will define a random inital policy. <br>
# Remember that a policy maps states to actions.

# In[10]:


policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)


# Next, we will randomly initialize the value fonction

# In[11]:


np.random.seed(1234)
# initialize V(s) - value function
V = {}
states = grid.all_states()
for s in states:
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# initial value for all states in grid
print_values(V, grid)


# Note that we set to Null the values of the terminal states. <br> 
# For the print_values() function to compile, we set them to 0.

# ### 2.2 Policy iteration - code completion
# 
# You will now have to complete the Policy iteration algorithm. <br>
# Remember that the algorithm works in two phases. <br>
# First, in the *policy evaluation* phase, the value function is update with the formula:
# 
# $$
# V^\pi(s) =  \sum_{s',a}  p(s'|s,\pi(s))(r + \gamma*V^\pi(s') 
# $$
# This part of the algorithm is already coded for you. <br>
# 
# Second, in the *policy improvement* step, the policy is updated with the formula:
# 
# $$
# \pi'(s) = \underset{a}{arg max}\big\{ \sum_{s',a}  p(s'|s,\pi(s))(r + \gamma*V^\pi(s') \big\}
# $$
# 
# This is the part of code you will have to complete. <br>
# 
# Note that in the current gridWorld, p(s'|s,a) is deterministic. <br>
# Run the algorithm until convergence.

# In[17]:


iteration=0
# repeat until the policy does not change
while True:
    print("values (iteration %d)" % iteration)
    print_values(V, grid)
    print("policy (iteration %d)" % iteration)
    print_policy(policy, grid)
    print('\n\n')

    # 1. policy evaluation step
    # this implementation does multiple policy-evaluation steps
    # this is different than in the algorithm from the slides 
    # which does a single one.
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a) # reward
                sprime = grid.current_state() # s' 
                V[s] = r + GAMMA * V[sprime]
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        if biggest_change < SMALL_ENOUGH:
            break

    #2. policy improvement step
    is_policy_converged = True
    for s in states:
        if s in policy:
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')
            # loop through all possible actions to find the best current action
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                sprime = grid.current_state() 
                v = r + GAMMA * V[sprime]
                if v > best_value:
                    best_value = v
                    new_a = a
            if new_a is None: 
                print('problem')
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

    if is_policy_converged:
        break
    iteration+=1


# Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.

# In[13]:


print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)

