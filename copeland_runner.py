from minizinc import Instance, Model, Solver
from iterative_copeland import *

class CopelandRunner:
    def __init__(self, model, solver, datafile, max=True):
        self.model = model
        self.solver = solver
        self.inst = Instance(self.solver, self.model)
        self.datafile = datafile
        self.rankings = []
        self.copeland_scores = []
        self.debug = False
        self.result = None
        self.max = max
        self.winner_set = []
    def run(self):
        try:
            self.inst.add_file(self.datafile)
        self.inst.add_string("old_solutions = [];")
        self.result = self.inst.solve(all_solutions=True)
        self.rankings = [self.result[i, "util_per_agent"] for i in range(len(self.result))]
        no_of_candidates = len(self.rankings)
        no_of_agents = len(self.rankings[0])
        pairwise_score = pairwiseScoreCalcListFull(self.rankings, no_of_candidates, no_of_agents)
        self.copeland_scores = copelandScoreFull(pairwise_score, no_of_candidates, no_of_agents)
        self.maximize() if self.max else self.minimize()
    def minimize(self):
        lowest_copeland_scores = min(self.copeland_scores)
        for idx in range(0, len(self.copeland_scores)):
            if lowest_copeland_scores == self.copeland_scores[idx]:
                self.winner_set.append(idx)
    def maximize(self):
        highest_copeland_scores = max(self.copeland_scores)
        for idx in range(0, len(self.copeland_scores)):
            if highest_copeland_scores == self.copeland_scores[idx]:
                self.winner_set.append(idx)
    def printWinningSolutions(self):
        [print("score:", self.result[i, "util_per_agent"],"position:",self.result[i, "position_of_agent"]) for i in self.winner_set]
            
    
if __name__ == "__main__":
    gecode = Solver.lookup("gecode")
    model = Model("minizinc/project_matching/project_matching.mzn")
    datafile = "" #"minizinc/photo_agents4.dzn"
    copeland_runner = CopelandRunner(model, gecode, datafile)
    copeland_runner.run()
    print(copeland_runner.copeland_scores)
    print(copeland_runner.winner_set)
    copeland_runner.printWinningSolutions()