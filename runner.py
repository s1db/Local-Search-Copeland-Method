from minizinc import Instance, Model, Solver
from iterative_copeland import *

'''
- [x] Create class.
- [ ] Create more test cases.
'''

gecode = Solver.lookup("gecode")

model = Model("minizinc/photo_agents.mzn")

instance = Instance(gecode, model)
instance.add_file("minizinc/photo_agents3.dzn")


result = instance.solve(all_solutions=True)
set_of_solutions = []
for i in range(len(result)):
    set_of_solutions.append(result[i, "util_per_agent"])

pairwise_score = pairwiseScoreCalcListFull(
    set_of_solutions, len(set_of_solutions), len(set_of_solutions[0]))
copeland_score = copelandScoreFull(pairwise_score, len(
    set_of_solutions), len(set_of_solutions[0]))

print(copeland_score)
lowest_copeland_scores = min(copeland_score)

# Since our aim is to maximize, the lowest copeland score is the copeland winner.

copeland_set = []
for idx in range(0, len(copeland_score)):
    if lowest_copeland_scores == copeland_score[idx]:
        copeland_set.append(idx)
for i in copeland_set:
    print(result[i])