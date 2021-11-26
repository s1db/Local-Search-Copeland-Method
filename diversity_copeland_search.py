from os import listdir
from os.path import isfile, join
import numpy as np
import iterative_copeland as ic
import deletion_copeland as dc
import random

from minizinc import Instance, Model, Result, Solver, Status

def diversityMaxCopeland(model, datafile):
    model_path = "./models/"+model+"/"+model+".mzn"
    print("path to model: ", model_path)
    # Initialize the model 
    m = Model(model_path) 
    # Solver and instance definition
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, m)
    instance.add_file("./models/" + model + "/data/" + datafile + ".dzn")
    save_at = model+"_profiles/"
    print("model and data initalized.")
    # this is tied to what we have in the "simple_diversity_mixin.mzn"
    variables_of_interest_key : str  = "diversity_variables_of_interest"

    # we'll need a solution pool of previously seen solutions
    # to rule out condorcet cycles; a solution is stored as a Python dictionary from variable to value

    # Flattended all solution pool, used for diversity calculation
    # all solutions seen till now
    all_solutions_seen = set()
    # utilities of all solutions seen till now
    all_solutions_utilities = []
    # solutions used for diversity constraints and deletions
    sol_pool  = []
    # Our utility preference profile
    pref_profile  = []
    search_more : bool = True
    no_solutions = 0
    restart = True
    seed = 0
    while search_more and no_solutions < 100:
        with instance.branch() as inst:
            if not restart: # once we have solutions, it makes sense to maximize diversity
                inst.add_string("solve maximize diversity_abs;")
                # removes the id from the solutions
                inst["old_solutions"] = np.stack(sol_pool, axis=0)[:,1:].flatten().tolist()
            else: # otherwise, we just aim to satisfy the constraints
                # inst.add_string("solve satisfy;")
                inst.add_string("solve :: int_search(diversity_variables_of_interest, input_order, indomain_random, complete) satisfy;")
                inst["old_solutions"] = []
                restart = False
            res = inst.solve(random_seed=seed)
            if res.solution is not None:
                if tuple(res["diversity_variables_of_interest"]) not in all_solutions_seen:
                    all_solutions_seen.add(tuple(res["diversity_variables_of_interest"]))
                    all_solutions_utilities.append(res["util_per_agent"])
                    # The solution obtained
                    sol_pool.append([no_solutions] + res["diversity_variables_of_interest"])
                    # The utility from solution obtained
                    pref_profile.append([no_solutions] + res["util_per_agent"])
                    no_solutions += 1
                else:
                    restart = True
                    seed = random.randint(0, 1000000)
            search_more = res.status in {Status.SATISFIED, Status.ALL_SOLUTIONS, Status.OPTIMAL_SOLUTION}
        # 10 solutions are in the pool
        if no_solutions % 11 == 0:
            pref_profile, copeland_score = dc.copelandWrapper(np.stack(pref_profile, axis=0), 5)
            not_deleted_ids = np.stack(pref_profile, axis=0)[:, 0].tolist()
            print(not_deleted_ids)
            print(no_solutions)
            sol_pool = [sol_pool[i] for i in range(len(sol_pool)) if sol_pool[i][0] in not_deleted_ids]
    print("all solutions seen: ", all_solutions_seen)
if __name__ == "__main__":
    benchmarks = ["project_assignment", "photo_placement"]
    benchmark = benchmarks[1]
    directory = "./models/"+benchmark+"/data"
    datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))][-2:-1]
    print(datafiles)
    for datafile in datafiles:
        diversityMaxCopeland(benchmark, datafile)
        
