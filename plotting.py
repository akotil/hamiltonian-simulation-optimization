import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import matrix_power

from calculations import Derivative
from simulation import Simulation
from unitary_optimization_demo import Optimization
from utils import build_whole_circuit

import seaborn as sns


def plot_error(N=6, T=1):
    K = [1]
    R = [2 ** x for x in range(7, 2, -1)]
    delta_t = [T / r for r in R]
    print(delta_t)
    colors = ["r", "b"]
    for k_idx, k in enumerate(K):
        trott_err = []
        opt_trot_err = []
        for t_idx, t in enumerate(delta_t):
            simulation = Simulation(N, T, R[t_idx], k * 2, False)
            exact_sol = simulation.Href
            trott_err.append(np.linalg.norm(simulation.trotterization - exact_sol))

            step_simulation = Simulation(N, t, 1, k * 2, False)

            print("non optimized step diff: ", np.linalg.norm(step_simulation.Href - step_simulation.trotterization))
            print("non optimized overall diff: ", np.linalg.norm(simulation.Href - simulation.trotterization))
            derivv = Derivative(step_simulation)
            optimization = Optimization(step_simulation, derivv)
            optimized_trotterization = optimization.optimize_trotterization(time_step=t, r=R[t_idx], recycle=True)

            print("optimized overall diff: ", np.linalg.norm(simulation.Href - optimized_trotterization))
            opt_trot_err.append(np.linalg.norm(optimized_trotterization - exact_sol))

        plt.plot(delta_t, trott_err, label="k=" + str(k), color=colors[k_idx])
        plt.plot(delta_t, opt_trot_err, label="optimized k=" + str(k), linestyle="dashed", color=colors[k_idx])

    plt.xlabel(r'$\delta t$')
    plt.ylabel("Error")

    plt.title(r'$N={}, T={}$'.format(N, T))

    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    plt.savefig("plots/final-ver")
    plt.show()


def compare_r(T, R):
    sns.set_theme(style='whitegrid')
    diff_opt = {}
    layer_arr = []
    for r in R:
        time_step = T / r
        file_name = "bins/time_step_{}_r_1_k_2".format(round(time_step, 4))
        no_layers = r * 3
        layer_arr.append(no_layers)
        print(time_step)
        sim = Simulation(N=6, T=T, r=r, order=2, periodic_boundary=False)
        circuit = build_whole_circuit(file_name, r)
        diff = np.linalg.norm(circuit - sim.Href)
        print(diff)
        diff_opt[r] = np.linalg.norm(circuit - sim.Href)

    diff_trott_1 = []
    r_1_arr = []
    last_min_r = 0
    for r in range(16, 150):
        min_r = 2 ** int(np.log2(r))
        if r in range(min_r, min_r + 10) and min_r != last_min_r:
            sim = Simulation(N=6, T=T, r=r, order=2, periodic_boundary=False)
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

    plt.scatter(r_1_arr, diff_trott_1, label="Trott. error: $||\delta_2-e^{-iHt}||$", marker="d")
    plt.scatter(R, diff_opt.values(), label="Opt. trott. error: $||\delta_2^{opt}-e^{-iHt}||$", marker="d")
    plt.xticks(r_1_arr + R, labels=[])

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
    plt.plot(diff_arr)
    plt.xlabel("Number of iterations")
    plt.ylabel("$||\delta_2^{opt}-e^{-i\Delta tH}||$")
    plt.xlim(right=len(diff_arr))
    plt.ylim(top=diff_arr[0])
    plt.title("$T=\Delta t=0.125s$")
    plt.savefig("plots/diff_arr", dpi=500)
    plt.show()
