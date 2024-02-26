import numpy as np

from utils import *

r1_before = np.loadtxt("gradient_conflict_losspred_1k_r1.csv", delimiter=",", skiprows=1)[:,1] / 100
r1_after = np.loadtxt("gradient_conflict_losspred_stopgrad_1k_r1.csv", delimiter=",", skiprows=1)[:,1] / 100
auroc_before = np.loadtxt("gradient_conflict_losspred_1k_auroc.csv", delimiter=",", skiprows=1)[:,1]
auroc_after = np.loadtxt("gradient_conflict_losspred_stopgrad_1k_auroc.csv", delimiter=",", skiprows=1)[:,1]
r1_before = r1_before[range(0, len(r1_after))]
auroc_before = auroc_before[range(0, len(auroc_after))]

less_wide_plot = bundles.icml2022(column="half")
less_wide_plot["figure.figsize"] = (2.8, less_wide_plot["figure.figsize"][1])

with plt.rc_context(less_wide_plot):
    fig, ax1 = plt.subplots()
    fig.set_figheight(1.8)

    # Plot the first line with its y-axis on the left
    plt.plot(range(1, len(r1_before) + 1), r1_before, color=RED)
    plt.text(len(r1_before) + 0.2, r1_before[len(r1_before) - 1] - 0.007, s="Accuracy (classifier)", va="top", ha="right",
             color=RED)
    plt.plot(range(1, len(auroc_before) + 1), auroc_before, color=BLUE)
    plt.text(len(auroc_before) + 0.2, auroc_before[len(auroc_before) - 1] + 0.004, s="AUROC (uncertainties)",
             va="bottom", ha="right", color=BLUE)
    ax1.set_xlabel('Train epoch')
    ax1.set_ylabel('Accuracy and AUROC')
    ax1.set_ylim((0.75, 0.83))
    plt.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")
    plt.xticks([1, 5, 10, 15, 20, 25, 30])
    plt.xlabel("Train epoch")
    plt.savefig("gradient_conflict_before.pdf")

with plt.rc_context(less_wide_plot):
    fig, ax1 = plt.subplots()
    fig.set_figheight(1.8)

    # Plot the first line with its y-axis on the left
    plt.plot(range(1, len(r1_after) + 1), r1_after, color=RED)
    plt.text(len(r1_after) + 0.2, r1_after[len(r1_after) - 1] - 0.008, s="Accuracy (classifier)", va="top",
             ha="right",
             color=RED)
    plt.plot(range(0, len(auroc_after) + 1), np.concatenate((np.zeros(1) + 0.73, auroc_after)), color=BLUE)
    plt.text(len(auroc_after) + 0.2, auroc_after[len(auroc_after) - 1] - 0.013, s="AUROC (uncertainties)",
             va="top", ha="right", color=BLUE)
    ax1.set_xlabel('Train epoch')
    ax1.set_ylabel('Accuracy and AUROC')
    ax1.set_ylim((0.75, 0.83))
    plt.grid(zorder=-1, color="lightgrey", lw=0.5, axis="y")

    plt.xticks([1, 5, 10, 15, 20, 25, 30])
    plt.xlabel("Train epoch")
    plt.savefig("gradient_conflict_after.pdf")
