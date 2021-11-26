"""
A minimal working example that shows how a diversity mixin would work.

In principle:
    1.) Get a solution
    2.) Define a boolean decision variable for every variable we care about
        Tie it to x[i] != x_old[i]
    3.) Set "maximize differences" as objective
    4.) Solve
"""

import logging
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import iterative_copeland as ic
import deletion_copeland as dc

logging.basicConfig(filename="minizinc-python.log", level=logging.DEBUG)

from minizinc import Instance, Model, Result, Solver, Status

def generatePreferenceProfile(model, datafile):
    model_path = "./models/"+model+"/"+model+".mzn"
    print(model_path) 
    m = Model(model_path) # "./models/photo_agents.mzn"
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, m)
    instance.add_file("./models/" + model + "/data/" + datafile + ".dzn")
    save_at = model+"_profiles/"

    print("Diversity maximization")

    # this is tied to what we have in the "simple_diversity_mixin.py"
    variables_of_interest_key : str  = "diversity_variables_of_interest"

    # we'll need a solution pool of previously seen solutions
    # to rule out condorcet cycles; a solution is stored as a Python dictionary from variable to value
    solution_pool = []
    all_sol_pool  = []
    pref_profile  = []
    search_more : bool = True
    no_solutions = 0
    while search_more:
        with instance.branch() as inst:
            if solution_pool: # once we have solutions, it makes sense to maximize diversity
                inst.add_string("solve maximize diversity_abs;")
            else:
                inst.add_string("solve satisfy;")

            inst["old_solutions"] = solution_pool
            res = inst.solve()

            if res.solution is not None:
                # print("utility:", res["util_per_agent"], "diversity:", res["diversity_variables_of_interest"])
                all_sol_pool.append(res["diversity_variables_of_interest"])
                pref_profile.append(res["util_per_agent"])
            search_more = res.status in {Status.SATISFIED, Status.ALL_SOLUTIONS, Status.OPTIMAL_SOLUTION}
            print(all_sol_pool)
            if search_more:
                next_sol_vars = res[variables_of_interest_key]  # copy the current solution variables
                solution_pool += next_sol_vars

    with open(save_at+'search_more'+datafile+'.vt', 'wb') as f:
        pickle.dump(np.array(all_sol_pool), f)
        pickle.dump(np.array(pref_profile), f)

if __name__ == "__main__":
    benchmarks = ["project_assignment", "photo_placement"]
    # for benchmark in benchmarks:
    #     directory = "./models/"+benchmark+"/data"
    #     datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))]
    #     print(datafiles)
    #     for datafile in datafiles:
    #         generatePreferenceProfile(benchmark, datafile)
    benchmark = benchmarks[1]
    directory = "./models/"+benchmark+"/data"
    datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))][:2]
    print(datafiles)
    for datafile in datafiles:
        generatePreferenceProfile(benchmark, datafile)