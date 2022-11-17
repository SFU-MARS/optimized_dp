# maximal Bipartite matching

class MaxMatching:
    def __init__(self, graph):
        """inputs
        graph: attackers and defenders, if attacker_i is defended by defender_j, then graph[i][j]=1
        """
        assert len(graph[0]) == len(graph[1]), 'The number of attackers and defenders should be the same!'
        self.graph = graph
        self.attackers = len(graph[0])  # the number of attackers
        self.defenders = len(graph[1])  # the number of defenders

    def match(self, attacker, matched, checked):
        # A DFS based recursive function is implemented
        # Return true if the attacker and defender match successfully
        for defender in range(self.defenders):
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

    def maxBPM(self):
        # matched[i] is the attacker defended by defender i, the value -1 indicates this attacker wins
        # in matched, the value is the number of attacker, index is the number of defender
        matched = [-1] * self.defenders
        # Count of the number of matched pairs
        result = 0
        for i in range(self.attackers):
            checked = [False] * self.defenders
            if self.match(i, matched, checked):
                result += 1
        # matched defenders
        matched_defenders = []
        matched_attackers = []
        matched_pairs = []
        for index in range(self.attackers):
            if matched[index] != -1:
                matched_attackers.append(matched[index]+1)
                matched_defenders.append(index+1)
                matched_pairs.append((matched[index]+1, index+1))  # (attacker, defender)
        for pair in zip(matched_defenders, matched_attackers):
            print(f'The attacker{pair[1]} is defended by the defender {pair[0]} \n')
        return result


# test
bpGraph = [[1, 1, 0, 1],
           [1, 0, 0, 1],
           [0, 1, 0, 0],
           [1, 0, 0, 0]]

g = MaxMatching(bpGraph)

print("Maximum matching number of this game is %d " % g.maxBPM())
