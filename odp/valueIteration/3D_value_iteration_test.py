import numpy as np
import time


###################################### USER-DEFINED FUNCTIONS ######################################


# return the successor states and their probabilities
def transition(state, action):
    return( 
        ( (0.1, state),
          (0.9, (state[0] + action[0], state[1] + action[1], state[2] + action[2]))
        ) 
    )

# return the reward for taking action from state
def reward(state, action):
    if state == (8,8,8): return 100
    else:                return 1



######################################### VALUE ITERATION ##########################################


# Solve for Vopt 
def solve_Vopt(Vopt, actions, gamma, epsilon, count):
    reSweep = True
    while (reSweep == True and count < 500):
        reSweep = False
        for i in range(0, Vopt.shape[0]):
            for j in range(0, Vopt.shape[1]):
                for k in range(0, Vopt.shape[2]):
                    oldV = Vopt[(i,j,k)]
                    newV = update_Vopt( (i,j,k), actions, Vopt, gamma )
                    if (abs(newV-oldV) > epsilon): reSweep = True
        count += 1
    return Vopt, count

# Update Vopt[(state)]
def update_Vopt(state, actions, Vopt, gamma):
    for action in actions:
        updatedV = reward(state, action)
        for probability, successor in transition(state, action):
            if (successor[0] < Vopt.shape[0]) and (successor[1] < Vopt.shape[1]) and (successor[2] < Vopt.shape[2]):
                updatedV += gamma * probability * Vopt[(successor)]
        if (Vopt[(state)] < updatedV):
            Vopt[(state)] = updatedV
    return Vopt[(state)]



############################################ INITIALIZE ############################################

Vopt    = np.zeros([30, 30, 30])
actions = ( (1,0,0), (0,1,0), (0,0,1) )
gamma   = 0.9
epsilon = 0.0000005
count   = 0 


############################################# EXECUTE ##############################################

t_s = time.time()
Vopt, count = solve_Vopt(Vopt, actions, gamma, epsilon, count)
t_e = time.time()

print(Vopt)
print()
print("Finished in ", count, " iterations, ")
print("Took        ", t_e-t_s, " seconds")
