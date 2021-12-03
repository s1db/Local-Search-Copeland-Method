from os import listdir
from os.path import isfile, join
import numpy as np
import iterative_copeland as ic
import deletion_copeland as dc
import random
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import os
from pathlib import Path

from minizinc import Instance, Model, Result, Solver, Status

from shutil import copyfile

debug_dir = ("debug")
debug = True

def create_debug_folder():
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

def log_and_debug_generated_files(child, model_counter):
    with child.files() as files:
        #print(files)
        if debug:
            # copy files to a dedicated debug folder
            for item in files[1:]:
                filename = os.path.basename(item)
                base, extension = os.path.splitext(filename)
                copyfile(item, os.path.join(debug_dir, f"mzn_condorcet_{model_counter}.{extension}"))

def diversityMaxCopeland(model, datafile, step, surviving_candidates, budget):
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
    timeout = timedelta(seconds=15)

    no_solutions = 0
    restart = True
    seed = 0
    copeland_score = []
    while search_more and no_solutions < budget:
        # print(no_solutions)
        with instance.branch() as inst:
            # disable already seen solutions
            for sol in all_solutions_seen:
                inst.add_string(f"constraint diversity_variables_of_interest != {list(sol)};\n")

            if not restart: # once we have solutions, it makes sense to maximize diversity
                inst.add_string("solve maximize diversity_abs;")
                # removes the id from the solutions
                solution_pool_mzn = np.stack(sol_pool, axis=0)[:,1:].tolist()
                #inst["old_solutions"] = solution_pool_mzn
                # UGLY workaround to avoid problems with too long jsons
                dec_vars_per_sol = len(sol_pool[0]) - 1
                s = "|\n".join([",".join(list(str(i) for i in l)) for l in solution_pool_mzn])
                s = f"[| {s} |]"
                inst.add_string(f"\nold_solutions = {s};\n")
                #all_solutions_mzn =  np.stack(list(all_solutions_seen), axis=0).flatten().tolist()
                #inst["all_seen_solutions"] = all_solutions_mzn
            else: # otherwise, we just aim to satisfy the constraints
                inst.add_string("solve satisfy;")
                # inst.add_string("solve :: int_search(diversity_variables_of_interest, input_order, indomain_random, complete) satisfy;")
                inst["old_solutions"] = []
                #inst["all_seen_solutions"] = []
                restart = False

            # , intermediate_solutions=True, all_solutions=True
            if debug:
                log_and_debug_generated_files(inst, no_solutions)

            call_succeeded = True
            try:
                #res = inst.solve(random_seed=seed, timeout = timeout, verbose=True, debug_output=Path(f"debug/debug_output{no_solutions}.txt"))
                res = inst.solve(random_seed=seed, timeout=timeout, optimisation_level=2)
            except Exception as e:
                print("❌ FAILED ❌")
                print(e)
                call_succeeded = False

            if call_succeeded and res.solution is not None:
                if tuple(res["diversity_variables_of_interest"]) not in all_solutions_seen:
                    all_solutions_seen.add(tuple(res["diversity_variables_of_interest"]))
                    all_solutions_utilities.append(res["util_per_agent"])
                    # The solution obtained
                    sol_pool.append([no_solutions] + res["diversity_variables_of_interest"])
                    # The utility from solution obtained
                    pref_profile.append([no_solutions] + res["util_per_agent"])
                    no_solutions += 1
                    # add a constraint that a solution cannot re-appear
                    next_sol = res["diversity_variables_of_interest"]
                else:
                    restart = True
                    seed = random.randint(0, 1000000)
                    print("Restarting with seed: ", seed)
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
    benchmarks = ["photo_placement_bipolar"] # "project_assignment", "photo_placement_bipolar",
    benchmarks = ["scheduling"]
    create_debug_folder()

    # benchmark = benchmarks[0]
    for benchmark in benchmarks:
        directory = "./models/"+benchmark+"/data"
        datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))]
        print(datafiles)
        datafile = datafiles[-1]
        pickled_file = open(benchmark + "_profiles/normal" + datafile+".vt", "rb")
        gt_pref_profile = pickle.load(pickled_file)
        gt_util_profile = pickle.load(pickled_file)
        gt_copeland_score = pickle.load(pickled_file)
        gt_copeland_score = [i/len(gt_copeland_score) for i in gt_copeland_score]
        try:
            solutions, copeland_scores = diversityMaxCopeland(benchmark, datafile, 100, 60, 150)
            #solutions, copeland_scores = diversityMaxCopeland(benchmark, datafile, 30, 10, 50)
            ids_in_complete_search = getID(np.array(solutions)[:,1:].tolist(), gt_pref_profile.tolist())
            if len(solutions) == len(copeland_scores):
                generatePlot(benchmark, datafile, gt_copeland_score, ids_in_complete_search, copeland_scores, True)
                dc.score_comparison(ids_in_complete_search, copeland_scores, gt_copeland_score)
            else:
                print(ids_in_complete_search)
                print(copeland_scores)
                print("NOT ENOUGH SOLUTIONS IN COMPLETE POOL")
        except Exception as e:
            print("TIMEOUT")
            print(e)