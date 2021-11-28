from os import listdir
from os.path import isfile, join
import numpy as np
import iterative_copeland as ic
import deletion_copeland as dc
import random
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta

from minizinc import Instance, Model, Result, Solver, Status

def diversityMaxCopeland(model, datafile, step, surviving_candidates, budget):
    model_path = "./models/"+model+"/"+model+".mzn"
    print("path to model: ", model_path)
    # Initialize the model 
    m = Model(model_path) 
    # Solver and instance definition
    gecode = Solver.lookup("chuffed")
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
    timeout = timedelta(seconds=15)

    no_solutions = 0
    restart = True
    seed = 0
    copeland_score = []
    while search_more and no_solutions <= budget:
        with instance.branch() as inst:
            if not restart: # once we have solutions, it makes sense to maximize diversity
                inst.add_string("solve maximize diversity_abs;")
                # removes the id from the solutions
                inst["old_solutions"] = np.stack(sol_pool, axis=0)[:,1:].flatten().tolist()
            else: # otherwise, we just aim to satisfy the constraints
                inst.add_string("solve satisfy;")
                #inst.add_string("solve :: int_search(diversity_variables_of_interest, input_order, indomain_random, complete) satisfy;")
                inst["old_solutions"] = []
                restart = False

            res = inst.solve(random_seed=seed, timeout = timeout)
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
                    # print("Restarting with seed: ", seed)
            search_more = res.status in {Status.SATISFIED, Status.ALL_SOLUTIONS, Status.OPTIMAL_SOLUTION}
        if no_solutions % 10 == 0:
            print(no_solutions)
        # 10 solutions are in the pool
        if len(sol_pool) == step:
            print("ran deletion")
            pref_profile, sol_pool, copeland_score = deletion(np.stack(pref_profile, axis=0), sol_pool, surviving_candidates)
    sol_pool = sol_pool[:surviving_candidates]
    copeland_score = [i/step for i in copeland_score]
    print(len(sol_pool))
    print(len(copeland_score))
    # Sanity Check
    assert len(sol_pool) == len(copeland_score)
    return sol_pool, copeland_score

def deletion(utility_profile_segment, sol_pool, surviving_candidates):
    # The preference profile is augmented with a column of candidate ids.
    agents = len(utility_profile_segment[0]) - 1
    candidates = len(utility_profile_segment)
    # calculation of pairwise score and copeland score
    # removes the ID column from the utility profile
    score_list = ic.pairwiseScoreCalcListFull(
        utility_profile_segment[:, 1:], candidates, agents)
    copeland_score = ic.copelandScoreFull(score_list, candidates, agents)

    # Finding the weaklings
    sorted_copeland_score = np.argsort(copeland_score)
    no_of_deleted_candidates = len(
        utility_profile_segment) - surviving_candidates
    # Takes the highest x candidates
    to_be_kept = sorted_copeland_score[no_of_deleted_candidates:]
    # sanity check
    assert len(sorted_copeland_score) == len(copeland_score) == utility_profile_segment.shape[0]
    # Deleting weaklings from from preference_profile, copeland_score and sol_pool.
    utility_profile_segment = [utility_profile_segment[x] for x in to_be_kept]
    copeland_score = [copeland_score[x] for x in to_be_kept]
    sol_pool = [sol_pool[x] for x in to_be_kept]
    # Sanity check
    assert len(utility_profile_segment) == len(copeland_score) == len(sol_pool)
    return (utility_profile_segment, sol_pool, copeland_score)

def getID(solutions, complete_profile):
    ids = []
    for solution in solutions:
        ids.append(complete_profile.index(solution))
    # Sanity check
    assert len(ids) == len(solutions)
    return ids

def generatePlot(directory, filename, true_copeland_score, not_deleted_candidate_id, cs, show_plot=True):
    winners_index = np.argsort(true_copeland_score)[-5:].tolist()
    winners_value = [true_copeland_score[x] for x in winners_index]
    true_copeland_score_deletion_labels = [true_copeland_score[i] for i in not_deleted_candidate_id]
    fig, (ax1) = plt.subplots(1)
    fig.suptitle(directory + ' ' + filename)
    fig.set_size_inches(11.69, 8.27)
    ax1.plot(true_copeland_score, 'b.')
    ax1.plot(not_deleted_candidate_id, cs, 'r.')
    ax1.plot(not_deleted_candidate_id, true_copeland_score_deletion_labels, 'm*')
    ax1.plot(winners_index, winners_value, 'y+')
    ax1.legend(['Copeland Score Post Deletion', 'Real Copeland Score'])
    ax1.set_xlabel('Candidate IDs')
    ax1.set_ylabel('Normalised Copeland Score')
    plt.savefig("plots/" + directory + "/png/diversity_search" + filename)
    if show_plot:
        plt.show()

if __name__ == "__main__":
    benchmarks = ["project_assignment", "photo_placement_bipolar"]
    benchmark = benchmarks[1]
    directory = "./models/"+benchmark+"/data"
    datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))][-2:-1]
    print(datafiles)
    for datafile in datafiles:
        pickled_file = open(benchmark + "_profiles/normal" + datafile+".vt", "rb")
        gt_pref_profile = pickle.load(pickled_file)
        gt_util_profile = pickle.load(pickled_file)
        gt_copeland_score = pickle.load(pickled_file)
        gt_copeland_score = [i/len(gt_copeland_score) for i in gt_copeland_score]
        solutions, copeland_scores = diversityMaxCopeland(benchmark, datafile, 60, 30, 80)
        ids_in_complete_search = getID(np.array(solutions)[:,1:].tolist(), gt_pref_profile.tolist())
        print(ids_in_complete_search)
        print(copeland_scores)
        generatePlot(benchmark, datafile, gt_copeland_score, ids_in_complete_search, copeland_scores, True)
        dc.score_comparison(ids_in_complete_search, copeland_scores, gt_copeland_score)