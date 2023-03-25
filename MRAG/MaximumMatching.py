# maximal Bipartite matching

class MaxMatching:
    def __init__(self, bp):
        """inputs
        bipartite graph: attackers and defenders, if attacker_i is defended by defender_j, then graph[i][j]=1
        """
        self.graph = bp
        self.num_attackers = len(bp)  # the number of attackers
        self.num_defenders = len(bp[0])  # the number of defenders

    def match(self, attacker, matched, checked):
        # A DFS based recursive function is implemented
        # Return true if the attacker and defender match successfully
        for defender in range(self.num_defenders):
            # If the edge between attacker and defender exists + this defender has not been checked before
            if self.graph[attacker][defender] and checked[defender] == False:
                # This defender has already been checked by this attacker
                checked[defender] = True
                # If this defender has not matched another attacker,
                # or this defender has matched another attacker but the attacker could match other defender
                if matched[defender] == -1 or self.match(matched[defender], matched, checked):
                    matched[defender] = attacker
                    return True
        return False

    def maximum_match(self):
        # matched[i] is the attacker defended by defender i, the value -1 indicates this attacker wins
        # in matched, the value is the number of attacker, index is the number of defender
        matched = [-1] * self.num_defenders
        # Count of the number of matched pairs
        result = 0
        for i in range(self.num_attackers):
            checked = [False] * self.num_defenders
            if self.match(i, matched, checked):
                # print(f"The checked list is {checked}")
                result += 1
        selected = [[] for _ in range(self.num_defenders)] # [[a1], [a2], ...]
        for j in range(self.num_defenders):
            if matched[j] != -1:
                selected[j].append(matched[j])
        return result, selected


# test
bpGraph = [[1, 1], 
           [0, 0], 
           [0, 0], 
           [1, 0]]


# bpGraph = np.array([[1, 1, 1], 
#                     [0, 1, 0], 
#                     [0, 1, 0],
#                     [1, 0, 0]])

# bpGraph = [[1, 1, 0, 1],
#            [1, 0, 0, 1],
#            [0, 1, 0, 0],
#            [1, 0, 0, 0]]


mm = MaxMatching(bpGraph)
number, selected = mm.maximum_match()
print(number)
print(selected)