import numpy as np
from utilities import *
from MaximumMatching import MaxMatching

# 
value1v1 = np.load('MRAG/1v1AttackDefend.npy')

# initialize positions of attackers and defenders
attackers_initials = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
defenders_initials = [(0.3, 0.5), (-0.3, 0.5)]

current_attackers = attackers_initials
current_defenders = defenders_initials

# maximum matching
bigraph = bi_graph(value1v1, current_attackers, current_defenders)
MaxMatch = MaxMatching(bigraph)
num, selected = MaxMatch.maximum_match()
print(f"The maximum matching pair number is {num} \n")
print(f"The result matching is {selected}")