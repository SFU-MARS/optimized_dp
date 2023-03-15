from utilities import *
from AttackerDefender1v1 import AttackerDefender1v1

# initialize positions of attackers and defenders
# demo: 4 attackers with 2 defenders
attackers0 = [(0.0, 0.0), (0.0, 0.8), (-0.5, 0.0), (0.5, -0.5)]
defenders0 = [(0.3, 0.5), (-0.3, 0.5)]
num_attacker = len(attackers0)
num_defender = len(defenders0)

attackers = [AttackerDefender1v1() for _ in range(num_attacker)]


# test: 1v1
value1v1 = np.load('MRAG/1v1AttackDefend.npy')

Ic = capture_individual(attackers0, defenders0, value1v1)
print(Ic)


# test 2v1:
value2v1 = np.load('MRAG/2v1AttackDefend.npy')

Pc = capture_pair(attackers0, defenders0, value2v1)
print(Pc)

selected = mip_solver(num_attacker=4, num_defender=2, Pc=Pc, Ic=Ic)
