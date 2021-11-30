"""
This script analyses the social welfare scores reported by our models to learn about
the distributions.

Executes the "standalone" versions that can also be used for debugging
"""

import logging
from os import listdir
from os.path import isfile, join
import os
import pickle

import imageio
import numpy as np
import iterative_copeland as ic
import matplotlib.pyplot as plt

logging.basicConfig(filename="model-analysis.log", level=logging.DEBUG)

from minizinc import Instance, Model, Result, Solver, Status

from celluloid import Camera
SOLVER = "gecode"
WELFARE = "social_welfare"
RUN_MZN = True # if false, just use pickled welfares

# search annotations for different traversals
NORMAL = "normal"
WORST_CASE = "worst_case"
BEST_CASE = "best_case"
RANDOM_WELFARES = "random_welfares"
RANDOM_DEC_VARS = "random_dec_vars"

def getWelfareDistribution(model, datafile, search_annot):
    model_file = "./models/"+model+"/"+model+"_standalone.mzn"
    print(model_file)
    m = Model(model_file) # "./models/photo_placement.mzn"

    solver = Solver.lookup(SOLVER)
    instance = Instance(solver, m)
    instance.add_file("./models/" + model + "/data/" + datafile + ".dzn")
    annot_str = "" if search_annot == NORMAL else f":: {search_annot}"
    instance.add_string(f"solve {annot_str} satisfy;")
    save_at = model+"_welfares/"

    if not os.path.exists(save_at):
        os.makedirs(save_at)
        os.makedirs(save_at + "/plots")

    try:
        # Find and print all intermediate solutions
        print(f"{search_annot} traversal")
        with instance.branch() as inst:
            result = inst.solve(all_solutions=True)

            welfares = []
            for i in range(len(result)):
                welfares.append(result[i, WELFARE])

            print("welfares", len(welfares))
            #print(welfares)

            # also pickle the welfares
            welfares = np.array(welfares)
            with open(save_at + search_annot + datafile +'.vt', 'wb') as f:
                pickle.dump(welfares, f)
            return welfares

    except Exception as e:
        print("❌ FAILED ❌")
        print(e)
        return None


def plot(welfares, model, data, search_annot, show_plots_during_execution=True ):
    save_at = model + "_welfares/plots/"

    # Reading pickled files and storing the data.
    fig, axs = plt.subplots(1, 2)
    plot_file = f"{model}_{data}_{search_annot}.png"
    fig.suptitle(f"{model} {data} {search_annot}")

    fig.set_size_inches(16.69, 8.27)
    axs[0].plot(welfares, 'b.')

    axs[0].set_xlabel('Candidate IDs')
    axs[0].set_ylabel('Social Welfare')

    # the histogram of the data
    values, counts = np.unique(welfares, return_counts=True)

    # then you plot away
    _ = axs[1].vlines(values, 0, counts, color='C0', lw=4)

    # optionally set y-axis up nicely
    #axs[1].ylim(0, max(counts) * 1.06)
    axs[1].set_xlabel("Social Welfare values")
    axs[1].set_ylabel("Frequency")

    plt.savefig(f"{save_at}{plot_file}")
    if show_plots_during_execution:
        plt.show()

def change_frame_rate(file):
    gif = imageio.mimread(file)
    imageio.mimsave(file, gif, fps=1)


def plot_gif(welfares, model, data, search_annot):
    # Reading pickled files and storing the data.
    save_at = model + "_welfares/plots/"

    # Reading pickled files and storing the data.
    plot_file = f"{save_at}{model}_{data}_{search_annot}.gif"

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"{model} {data} {search_annot}")

    camera = Camera(fig)
    for i in range(3, len(welfares)):
        fig.suptitle(f"{model} {data} {search_annot}")
        fig.set_size_inches(16.69, 8.27)
        axs[0].plot(welfares[:i], 'b.')

        axs[0].set_xlabel('Candidate IDs')
        axs[0].set_ylabel('Social Welfare')

        # the histogram of the data
        values, counts = np.unique(welfares[:i], return_counts=True)

        # then you plot away
        _ = axs[1].vlines(values, 0, counts, color='C0', lw=4)

        # optionally set y-axis up nicely
        # axs[1].ylim(0, max(counts) * 1.06)
        axs[1].set_xlabel("Social Welfare values")
        axs[1].set_ylabel("Frequency")

        camera.snap()
        # ax1.lines.pop(0)
    animation = camera.animate(blit=False)

    animation.save(plot_file, writer='imagemagick')
    #change_frame_rate(plot_file)

if __name__ == "__main__":
    models = ["project_assignment"] #, "project_assignment", "photo_placement_bipolar"
    models = ["photo_placement_bipolar", "project_assignment", "vehicle_routing", "scheduling"]
    models = ["scheduling"]
    #models = ["vehicle_routing"]

    for model in models:
        directory = f"./models/{model}/data"
        datafiles = [f[:-4] for f in listdir(directory) if isfile(join(directory, f))]
        print(datafiles)
        for datafile in datafiles[2:] :   #[:1]: # just the first for now:
            print(datafile)

            #for annot in [NORMAL]:
            for annot in [NORMAL, WORST_CASE, BEST_CASE, RANDOM_WELFARES, RANDOM_DEC_VARS]:
                welfares = getWelfareDistribution(model, datafile, annot)
                plot(welfares, model, datafile, annot, show_plots_during_execution=False)
                #plot_gif(welfares, model, datafile, annot)