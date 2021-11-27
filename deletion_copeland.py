from numpy.core.fromnumeric import sort
import iterative_copeland as ic
import pickle
import numpy as np
from celluloid import Camera
import imageio

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

def copelandWrapper(utility_profile_segment, surviving_candidates):
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
    # Deleting weaklings from from preference profile and copeland score.
    utility_profile_segment = [utility_profile_segment[x] for x in to_be_kept]
    copeland_score = [copeland_score[x] for x in to_be_kept]
    # Sanity check
    assert len(utility_profile_segment) == len(copeland_score)
    return (utility_profile_segment, copeland_score)

def score_comparison(indexes, deletion_copeland_scores, copeland_scores):
    deletion_winner_cs = max(deletion_copeland_scores) # Copeland winner of deletion algorithm
    id_of_deletion_winner = indexes[deletion_copeland_scores.index(deletion_winner_cs)] # ID of Copeland winner of deletion algorithm 
    ground_truth_score_of_deletion_winner = copeland_scores[id_of_deletion_winner] # Ground Truth Copeland Score of defined ID
    ground_truth_winner_cs = max(copeland_scores) # Ground Truth Copeland Winner
    print("        Reliability: GT Winner vs Deletion Winner(on GT)       :"+ str(ground_truth_winner_cs / ground_truth_score_of_deletion_winner))
    print("        Gap:         Deletion Winner(on GT) vs Deletion Winner :"+ str(abs(deletion_winner_cs - ground_truth_score_of_deletion_winner) / ground_truth_score_of_deletion_winner))

def deletionCopeland(utility_profile, step, surviving_candidates, budget):
    # Processing the data.
    candidates = len(utility_profile)
    agents = len(utility_profile[0])
    i_utility_profile = None
    copeland_score = None
    for i in range(0, budget, step):
        # Growing set of candidiates.
        if i == 0:
            i_utility_profile = utility_profile[0:step]
        else: 
            i_utility_profile = np.append(
                i_utility_profile, utility_profile[i: i+step], axis=0)
        i_utility_profile, copeland_score = copelandWrapper(
            i_utility_profile, surviving_candidates)
    # Sanity check
    assert i_utility_profile != None and copeland_score != None
    return (i_utility_profile, copeland_score)

def deletionCopelandFamily(utility_profile, step, surviving_candidates, budget):
    # Processing the data.
    candidates = len(utility_profile)
    # We add a column on the 0th index to keep track of candidate IDs thus, we remove that column.
    agents = len(utility_profile[0]) - 1
    familyipp = []
    familycs = []
    i_utility_profile = None
    for i in range(0, budget, step):
        if i == 0:
            i_utility_profile = utility_profile[0:step]
        else: 
            i_utility_profile = np.append(
                i_utility_profile, utility_profile[i: i+step], axis=0)
        i_utility_profile, copeland_score = copelandWrapper(
            i_utility_profile, surviving_candidates)
        # Storing preference profile and associated copeland score
        familyipp.append(i_utility_profile)
        familycs.append(copeland_score)
    assert len(familyipp) == len(familycs)
    for i in range(len(familyipp)):
        assert len(familyipp[i]) == len(familycs[i])
    return (familyipp, familycs)


def plot(directory, filename, step, surviving_candidates, budget, show_plots_during_execution):
    # Reading pickled files and storing the data.
    pickled_file = open(directory + "_profiles/" + filename+".vt", "rb")
    preference_profile = pickle.load(pickled_file)
    utility_profile = pickle.load(pickled_file)
    true_copeland_score = None
    # Processing the data.
    candidates = len(utility_profile)
    agents = len(utility_profile[0])

    # Augmenting preference profile with IDs.
    utility_profile = np.insert(
        utility_profile, 0, range(candidates), axis=1)

    score_list = ic.pairwiseScoreCalcListFull(
        utility_profile[:, 1:], candidates, agents)
    true_copeland_score = ic.copelandScoreFull(score_list, candidates, agents)
    # Relative Copeland Score
    true_copeland_score = [i/candidates for i in true_copeland_score]
    pickled_file.close()
    ipp, cs = deletionCopeland(
        utility_profile, step, surviving_candidates,budget)
    cs = [i/(step+surviving_candidates) for i in cs]
    not_deleted_candidate_ids = np.stack(ipp, axis=0)[:, 0].tolist()
    winners_index = np.argsort(true_copeland_score)[-5:].tolist()
    winners_value = [true_copeland_score[x] for x in winners_index]
    true_copeland_score_deletion_labels = [true_copeland_score[i] for i in not_deleted_candidate_ids]    
    score_comparison(not_deleted_candidate_ids, cs, true_copeland_score)
    
    fig, (ax1) = plt.subplots(1)
    fig.suptitle(directory + ' ' + filename)
    fig.set_size_inches(11.69, 8.27)
    ax1.plot(true_copeland_score, 'b.')
    ax1.plot(not_deleted_candidate_ids, cs, 'ro')
    ax1.plot(not_deleted_candidate_ids, true_copeland_score_deletion_labels, 'm*')
    ax1.plot(winners_index, winners_value, 'y+')
    ax1.legend(['Copeland Score Post Deletion', 'Real Copeland Score'])
    ax1.set_xlabel('Candidate IDs')
    ax1.set_ylabel('Normalised Copeland Score')
    plt.savefig("plots/" + directory + "/png/" + filename)
    if show_plots_during_execution:
        plt.show()


def plot_gif(directory, filename, step, surviving_candidates, budget):
    # Reading pickled files and storing the data.
    pickled_file = open(directory + "_profiles/" + filename+".vt", "rb")

    true_copeland_score = None
    preference_profile = pickle.load(pickled_file)
    utility_profile = pickle.load(pickled_file)
    candidates = len(utility_profile)
    agents = len(utility_profile[0])
    # Augmenting preference profile with IDs.
    utility_profile = np.insert(
        utility_profile, 0, range(candidates), axis=1)

    score_list = ic.pairwiseScoreCalcListFull(
        utility_profile[:, 1:], candidates, agents)
    true_copeland_score = ic.copelandScoreFull(score_list, candidates, agents)
    true_copeland_score = [i/candidates for i in true_copeland_score]
    pickled_file.close()

    familyipp, familycs = deletionCopelandFamily(
        utility_profile, step, surviving_candidates,budget)
    winners_index = np.argsort(true_copeland_score)[-5:].tolist()
    winners_value = [true_copeland_score[x] for x in winners_index]

    fig, ax1 = plt.subplots()
    camera = Camera(fig)
    for i in range(len(familyipp)):
        not_deleted_candidate_id = np.stack(
            familyipp[i], axis=0)[:, 0].tolist()
        true_copeland_score_deletion_labels = [true_copeland_score[i] for i in not_deleted_candidate_id]
        # print(not_deleted_candidate_id)
        cs = familycs[i]
        cs = [j/(step+surviving_candidates) for j in cs]
        assert len(cs) == len(not_deleted_candidate_id)
        fig.suptitle(directory + ' ' + filename)
        fig.set_size_inches(11.69, 8.27)
        ax1.set_ylim(-0.1, 1.1)
        ax1.plot(true_copeland_score, 'bo')
        ax1.plot(winners_index, winners_value, 'y+')
        ax1.plot(not_deleted_candidate_id, cs, 'ro')
        ax1.plot(not_deleted_candidate_id, true_copeland_score_deletion_labels, 'm*')
        ax1.legend(['Real Copeland Score', 'Copeland Score Post Deletion'])
        ax1.set_xlabel('Candidate IDs')
        ax1.set_ylabel('Normalised Copeland Score')
        camera.snap()
        # ax1.lines.pop(0)
    animation = camera.animate(blit=False)
    filepath = "plots/" + directory + "/gif/" + filename + '.gif'
    animation.save(filepath, writer='imagemagick')
    change_frame_rate(filepath)


def change_frame_rate(file):
    gif = imageio.mimread(file)
    imageio.mimsave(file, gif, fps=1)


if __name__ == "__main__":
    SHOW_PLOTS_DURING_EXECUTION = False
    PLOT_GIFS = True
    benchmarks = [ "photo_placement_bipolar"]
    step = 100
    surviving_candidates = 50
    budget = 500
    profile_types = ["inverted", "normal"] # "random", "search_more"
    for benchmark in benchmarks:
        print("🟢 Running " + benchmark)
        for i in range(1,10):  # , '1','2','3','4', '5', '6'
            for profile_type in profile_types:
                try:
                    print("    " + profile_type+str(i))
                    plot(benchmark, profile_type+str(i), step,
                            surviving_candidates, budget, SHOW_PLOTS_DURING_EXECUTION)
                    # plot_gif(benchmark, profile_type+str(i),
                    #             step, surviving_candidates, budget)
                except Exception as e:
                    print(e)
                    None