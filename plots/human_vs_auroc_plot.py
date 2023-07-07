import matplotlib.pyplot as plt

from plots_constants import *
from scipy.stats import spearmanr

with plt.rc_context(bundles.neurips2023()):
    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.set_figheight(2.4)
    for x, y, color, marker, size in zip(res["best_furthertest_avg_downstream_auroc_correct"].to_list(),
                                         res["best_furthertest_avg_downstream_rcorr_entropy"].to_list(),
                                         res["color"].to_list(),
                                         res["marker"].to_list(),
                                         res["size"].to_list()):
        ax.scatter(x,
                   y,
                   marker=marker,
                   c=color,
                   s=size,
                   zorder=2)
    ax.set_xlabel("Downstream R-AUROC")
    ax.set_ylabel("Alignment with Human Uncertainties")
    rankcorr = spearmanr(res["best_furthertest_avg_downstream_auroc_correct"].to_numpy(),
                         res["best_furthertest_avg_downstream_rcorr_entropy"].to_numpy(), nan_policy="omit")[0]
    ax.text(.02, .98, f'Rank Correlation = {rankcorr:.2f}', ha='left', va='top', transform=ax.transAxes)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)

    for x, y, color, marker, size in zip(res["best_furthertest_avg_downstream_auroc_correct"].to_list(),
                                         res["best_furthertest_avg_downstream_croppedHasBiggerUnc"].to_list(),
                                         res["color"].to_list(),
                                         res["marker"].to_list(),
                                         res["size"].to_list()):
        ax2.scatter(x,
                   y,
                   marker=marker,
                   c=color,
                   s=size,
                   zorder=2)
    #plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    ax2.set_xlabel("Downstream R-AUROC")
    ax2.set_ylabel("Pct. Cropped Image has Higher Uncertainty")
    rankcorr = spearmanr(res["best_furthertest_avg_downstream_auroc_correct"].to_numpy(),
                         res["best_furthertest_avg_downstream_croppedHasBiggerUnc"].to_numpy(), nan_policy="omit")[0]
    ax2.text(.02, .98, f'Rank Correlation = {rankcorr:.2f}', ha='left', va='top', transform=ax2.transAxes)
    ax2.grid(zorder=-1, color="lightgrey", lw=0.5)

    plt.savefig("human.pdf")
    fig.set_figheight(1.9)
    fig.set_figwidth(7)
    plt.savefig("human_poster.png", dpi=1200)
    plt.close()
