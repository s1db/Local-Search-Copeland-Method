'''
TODO:
    - [ ] Create a wrapper for score_list and copeland_score. 
    - [ ] Put deletion step as a function.
'''

import iterative_copeland as ic
import pickle
import numpy as np
from celluloid import Camera
import imageio

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

def copelandWrapper(preference_profile_segment):
    agents = len(preference_profile_segment[0]) - 1
    candidates = len(preference_profile_segment)
    # calculation of pairwise score and copeland score
    score_list = ic.pairwiseScoreCalcListFull(
        preference_profile_segment[:, 1:], candidates, agents)
    copeland_score = ic.copelandScoreFull(score_list, candidates, agents)

    # Finding the weaklings
    sorted_copeland_score = np.argsort(copeland_score)
    no_of_deleted_candidates = len(
        preference_profile_segment) - surviving_candidates
    # Takes the highest x candidates
    to_be_deleted = sorted_copeland_score[0:no_of_deleted_candidates]

    preference_profile_segment = [i for x, i in enumerate(
        preference_profile_segment) if x not in to_be_deleted]
    copeland_score = [i for x, i in enumerate(
        copeland_score) if x not in to_be_deleted]
    assert len(preference_profile_segment) == len(copeland_score)
    # # Recomputing copeland score
    # score_list = ic.pairwiseScoreCalcListFull(
    #     preference_profile_segment, len(preference_profile_segment), agents)
    # copeland_score = ic.copelandScoreFull(
    #     score_list, len(preference_profile_segment), agents)
    return (preference_profile_segment, copeland_score)


def deletionCopeland(preference_profile, step, surviving_candidates):
    # Processing the data.
    candidates = len(preference_profile)
    agents = len(preference_profile[0])
    i_preference_profile = []
    for i in range(0, candidates, step):
        # Growing set of candidiates.
        i_preference_profile.extend(preference_profile[i: i+step])
        score_list = ic.pairwiseScoreCalcListFull(
            i_preference_profile, len(i_preference_profile), agents)
        copeland_score = ic.copelandScoreFull(
            score_list, len(i_preference_profile), agents)

        sorted_copeland_score = np.argsort(copeland_score)
        no_of_deleted_candidates = len(
            i_preference_profile) - surviving_candidates

        to_be_deleted = sorted_copeland_score[0:no_of_deleted_candidates]
        i_preference_profile = [i for j, i in enumerate(
            i_preference_profile) if j not in to_be_deleted]
    score_list = ic.pairwiseScoreCalcListFull(
        i_preference_profile, len(i_preference_profile), agents)
    copeland_score = ic.copelandScoreFull(
        score_list, len(i_preference_profile), agents)
    return (i_preference_profile, copeland_score)

def deletionCopelandFamily(preference_profile, step, surviving_candidates):
    # Processing the data.
    candidates = len(preference_profile)
    # We add a column on the 0th index to keep track of candidate IDs.
    agents = len(preference_profile[0]) - 1
    familyipp = []
    familycs = []
    i_preference_profile = preference_profile[0:step]
    for i in range(0, candidates, step):
        i_preference_profile, copeland_score = copelandWrapper(
            i_preference_profile)
        # Storing preference profile and associated copeland score
        familyipp.append(i_preference_profile)
        familycs.append(copeland_score)
        # step-size equivalent no of candidates added.
        i_preference_profile = np.append(
            i_preference_profile, preference_profile[i: i+step], axis=0)
    i_preference_profile, copeland_score = copelandWrapper(
        i_preference_profile)
    familyipp.append(i_preference_profile)
    familycs.append(copeland_score)
    assert len(familyipp) == len(familycs)
    for i in range(len(familyipp)):
        # print(len(familyipp[i]))
        # print(len(familycs[i]))
        # print("⭐")
        assert len(familyipp[i]) == len(familycs[i])
    return (familyipp, familycs)


def plot(directory, filename, step, surviving_candidates, show_plots_during_execution):
    # Reading pickled files and storing the data.
    pickled_file = open(directory + "_profiles/" + filename+".vt", "rb")
    preference_profile = pickle.load(pickled_file)
    true_copeland_score = None
    # Processing the data.
    candidates = len(preference_profile)
    agents = len(preference_profile[0])

    score_list = ic.pairwiseScoreCalcListFull(
        preference_profile, candidates, agents)
    true_copeland_score = ic.copelandScoreFull(score_list, candidates, agents)
    # Relative Copeland Score
    true_copeland_score = [i/candidates for i in true_copeland_score]
    pickled_file.close()

    ipp, cs = deletionCopeland(
        preference_profile, step, surviving_candidates)
    cs = [i/len(cs) for i in cs]
    not_deleted_candidate_ids = []
    for i, x in enumerate(preference_profile.tolist()):
        if x in np.array(ipp).tolist():
            not_deleted_candidate_ids.append(i)

    fig, (ax1) = plt.subplots(1)
    fig.suptitle('Copeland Scores')
    fig.set_size_inches(11.69, 8.27)
    ax1.plot(not_deleted_candidate_ids, cs, 'ro')
    ax1.plot(range(candidates), true_copeland_score, 'bo')
    # encircled_true_copeland_scores = []
    # ax1.plot(, true_copeland_score, 'ro')
    ax1.legend(['Copeland Score Post Deletion', 'Real Copeland Score'])
    ax1.set_xlabel('Candidate IDs')
    ax1.set_ylabel('Normalised Copeland Score')
    plt.savefig("plots/" + directory + "/png/" + filename)
    if show_plots_during_execution:
        plt.show()


def plot_gif(directory, filename, step, surviving_candidates):
    # Reading pickled files and storing the data.
    pickled_file = open(directory + "_profiles/" + filename+".vt", "rb")

    true_copeland_score = None
    preference_profile = pickle.load(pickled_file)
    candidates = len(preference_profile)
    agents = len(preference_profile[0])
    # Augmenting preference profile with IDs.
    preference_profile = np.insert(
        preference_profile, 0, range(candidates), axis=1)

    score_list = ic.pairwiseScoreCalcListFull(
        preference_profile[:, 1:], candidates, agents)
    true_copeland_score = ic.copelandScoreFull(score_list, candidates, agents)
    true_copeland_score = [i/candidates for i in true_copeland_score]
    pickled_file.close()

    familyipp, familycs = deletionCopelandFamily(
        preference_profile, step, surviving_candidates)
    winners_index = np.argsort(true_copeland_score)[-5:].tolist()
    winners_value = [true_copeland_score[x] for x in winners_index]
    assert len(familyipp) == len(familycs)
    for i in range(len(familycs)):
        assert len(familyipp[i]) == len(familycs[i])
    fig, ax1 = plt.subplots()
    camera = Camera(fig)
    for i in range(len(familyipp)):
        not_deleted_candidate_id = np.stack(
            familyipp[i], axis=0)[:, 0].tolist()
        cs = familycs[i]
        cs = [j/len(cs) for j in cs]
        assert len(cs) == len(not_deleted_candidate_id)
        fig.suptitle('Copeland Scores')
        fig.set_size_inches(11.69, 8.27)
        ax1.set_ylim(-0.1, 1.1)
        ax1.plot(range(candidates), true_copeland_score, 'bo')
        ax1.plot(winners_index, winners_value, 'm*')
        ax1.legend(['Real Copeland Score', 'Copeland Score Post Deletion'])
        ax1.set_xlabel('Candidate IDs')
        ax1.set_ylabel('Normalised Copeland Score')
        ax1.plot(not_deleted_candidate_id, cs, 'ro')
        camera.snap()
        # ax1.lines.pop(0)
    animation = camera.animate(blit=False)
    animation.save("plots/" + directory + "/gif/" +
                   filename + '.gif', writer='imagemagick')


def plot_copeland_winner(directory, filename, step, surviving_candidates, winner_id):
    # Reading pickled files and storing the data.
    pickled_file = open(directory + "_profiles/" + filename+".vt", "rb")
    preference_profile = pickle.load(pickled_file)
    true_copeland_score = None
    # Processing the data.
    candidates = len(preference_profile)
    agents = len(preference_profile[0])

    score_list = ic.pairwiseScoreCalcListFull(
        preference_profile, candidates, agents)
    true_copeland_score = ic.copelandScoreFull(score_list, candidates, agents)
    true_copeland_score = [i/candidates for i in true_copeland_score]
    pickled_file.close()

    familyipp, familycs = deletionCopelandFamily(
        preference_profile, step, surviving_candidates)

    assert len(familyipp) == len(familycs)
    for i in range(len(familycs)):
        assert len(familyipp[i]) == len(familycs[i])
    winner_copeland_score = []
    fig, ax1 = plt.subplots()
    for i in range(len(familyipp)):
        not_deleted_candidate_ids = familyndci[i]
        cs = familycs[i]
        cs = [j/len(cs) for j in cs]
        fig.suptitle('Copeland Scores')
        fig.set_size_inches(11.69, 8.27)
        ax1.plot(range(candidates), true_copeland_score, 'bo')
        ax1.legend(['Real Copeland Score', 'Overall True Copeland Winners',
                   'Copeland Score Post Deletion'])
        ax1.set_xlabel('Candidate IDs')
        ax1.set_ylabel('Normalised Copeland Score')
        ax1.plot(not_deleted_candidate_ids, cs, 'ro')
        # ax1.lines.pop(0)


def change_frame_rate(benchmark):
    directory = "./plots/" + benchmark + "/gif/"
    onlyfiles = [directory +
                 f for f in listdir(directory) if isfile(join(directory, f))]
    # print(onlyfiles)
    for file in onlyfiles:
        gif = imageio.mimread(file)
        imageio.mimsave(file, gif, fps=1)


if __name__ == "__main__":
    SHOW_PLOTS_DURING_EXECUTION = False
    PLOT_GIFS = True
    benchmarks = ["project_assignment", "photo_placement"]
    step = 20
    surviving_candidates = 7
    profile_types = ["inverted", "normal", "random", "search_more"]
    for benchmark in benchmarks:
        print("🟢 Running " + benchmark)
        for profile_type in profile_types:
            for i in ['1', '2', '3', '4', '5', '6']:  # , '1','2','3','4', '5', '6'
                try:
                    plot(benchmark, profile_type+str(i), step,
                         surviving_candidates, SHOW_PLOTS_DURING_EXECUTION)
                    plot_gif(benchmark, profile_type+str(i),
                             step, surviving_candidates)
                    print("    " + profile_type+str(i))
                except Exception as e:
                    print(e)
                    None
        change_frame_rate(benchmark)
