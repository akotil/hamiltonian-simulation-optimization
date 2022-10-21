import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_power

from simulation import Simulation
from utils import build_circuit_from_pickle
import seaborn as sns


def compare_r(T, R):
    sns.set_theme(style='whitegrid')
    plt.rcParams['xtick.bottom'] = True
    diff_opt = {}
    periodic = True
    for r in R:
        time_step = T / r
        file_name = "bins/time_step_{}_r_1_k_2".format(round(time_step, 4))
        if periodic:
            file_name += "_periodic"
        sim = Simulation(N=6, T=T, r=r, order=2, periodic_boundary=periodic)
        circuit = build_circuit_from_pickle(file_name, r, periodic=periodic, N=6)
        diff_opt[r] = np.linalg.norm(circuit - sim.Href)

    diff_trott_1 = []
    r_1_arr = []
    last_min_r = 0
    for r in range(16, 150):
        min_r = 2 ** int(np.log2(r))
        if r in range(min_r, min_r + 10) and min_r != last_min_r:
            sim = Simulation(N=6, T=T, r=r, order=2, periodic_boundary=periodic)
            trott_diff = np.linalg.norm(sim.trotterization - sim.Href)
            if trott_diff <= diff_opt[min_r]:
                diff_trott_1.append(trott_diff)
                r_1_arr.append(r)
                plt.axvspan(min_r, r, color='red', alpha=0.1)
                last_min_r = min_r
                if min_r == R[0]:
                    text_x = min_r - 12
                else:
                    text_x = r + 2
                plt.text(text_x, trott_diff, "$\Delta r={}$".format(r - min_r), fontsize="small")
                print(r)

    plt.scatter(r_1_arr, diff_trott_1, label="Trott. error: $||\delta_2-e^{-iHt}||$", marker="d")
    plt.scatter(R, diff_opt.values(), label="Opt. trott. error: $||\delta_2^{opt}-e^{-iHt}||$", marker="d")
    plt.xticks(r_1_arr, labels=[str(r) for r in r_1_arr])
    plt.xticks(R, labels=[str(r) for r in R])

    plt.xlabel("$r$")
    plt.ylabel("Error")

    # plt.xscale("log")
    plt.yscale("log")

    plt.legend(loc="upper right")
    plt.savefig("plots/comparison", dpi=500)

    plt.show()


def plot_diff_arr_from_pickle(file_name):
    diff_arr = pickle.load(open(file_name, "rb"))
    sns.set(rc={"axes.grid": True, "xtick.bottom": True, "axes.facecolor": "None",
                "axes.edgecolor": "lightgrey"})
    sns.set_theme(style='whitegrid', palette='deep')
    plt.plot(diff_arr, label="$k=1$")
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("$||\delta_2^{opt}-e^{-i\Delta tH}||$")
    plt.xlim(right=len(diff_arr))
    plt.ylim(top=diff_arr[0])
    plt.title("$T=\Delta t=2^{-7}s$")
    plt.savefig("plots/diff_arr", dpi=500)
    plt.show()
