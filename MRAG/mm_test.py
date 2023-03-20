# Python program to find
# maximal Bipartite matching.

class GFG:
	def __init__(self,graph):
		
		# residual graph
		self.graph = graph
		self.ppl = len(graph)
		self.jobs = len(graph[0])

	# A DFS based recursive function
	# that returns true if a matching
	# for vertex u is possible
	def bpm(self, u, matchR, seen):

		# Try every job one by one
		for v in range(self.jobs):

			# If applicant u is interested
			# in job v and v is not seen
			if self.graph[u][v] and seen[v] == False:
				
				# Mark v as visited
				seen[v] = True

				'''If job 'v' is not assigned to
				an applicant OR previously assigned
				applicant for job v (which is matchR[v])
				has an alternate job available.
				Since v is marked as visited in the
				above line, matchR[v] in the following
				recursive call will not get job 'v' again'''
				if matchR[v] == -1 or self.bpm(matchR[v],
											matchR, seen):
					matchR[v] = u
					return True
		return False

	# Returns maximum number of matching
	def maxBPM(self):
		'''An array to keep track of the
		applicants assigned to jobs.
		The value of matchR[i] is the
		applicant number assigned to job i,
		the value -1 indicates nobody is assigned.'''
		matchR = [-1] * self.jobs
		
		# Count of jobs assigned to applicants
		result = 0
		for i in range(self.ppl):
			seen = [False] * self.jobs
			if self.bpm(i, matchR, seen):
				result += 1
		return result



bpGraph = [[1, 1, 1],
           [1, 0, 1],
           [0, 1, 0],
           [1, 0, 0]]
# bpGraph =[[0, 1, 1, 0, 0, 0],
# 		[0, 0, 0, 0, 0, 1]]

print(f"The number of attackers is {len(bpGraph)}")
print(f"The number of defenders is {len(bpGraph[0])}")


g = GFG(bpGraph)


print ("Maximum number of applicants that can get job is %d " % g.maxBPM())

# This code is contributed by Neelam Yadav
