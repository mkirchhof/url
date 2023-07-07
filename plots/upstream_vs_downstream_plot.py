from plots_constants import *
from scipy.stats import spearmanr

with plt.rc_context(bundles.neurips2023()):
    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.set_figheight(2.4)
    for x, y, color, marker, size in zip(res["best_eval_auroc_correct"].to_list(),
                                         res["best_test_avg_downstream_auroc_correct"].to_list(),
                                         res["color"].to_list(),
                                         res["marker"].to_list(),
                                         res["size"].to_list()):
        ax.scatter(x,
                   y,
                   marker=marker,
                   c=color,
                   s=size,
                   zorder=2)
    ax.set_xlabel("Upstream R-AUROC")
    ax.set_ylabel("Downstream R-AUROC")
    rankcorr = spearmanr(res["best_eval_auroc_correct"].to_numpy(),
                         res["best_test_avg_downstream_auroc_correct"].to_numpy(), nan_policy="omit")[0]
    ax.text(.02, .98, f'Rank Correlation = {rankcorr:.2f}', ha='left', va='top', transform=ax.transAxes)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)

    for x, y, color, marker, size in zip(res["best_eval_croppedHasBiggerUnc"].to_list(),
                                         res["best_test_avg_downstream_auroc_correct"].to_list(),
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
    ax2.set_xlabel("Upstream Pct. Cropped Image has Higher Uncertainty")
    ax2.set_ylabel("Downstream R-AUROC")
    rankcorr = spearmanr(res["best_eval_croppedHasBiggerUnc"].to_numpy(),
                         res["best_test_avg_downstream_auroc_correct"].to_numpy(), nan_policy="omit")[0]
    ax2.text(.02, .98, f'Rank Correlation = {rankcorr:.2f}', ha='left', va='top', transform=ax2.transAxes)
    ax2.grid(zorder=-1, color="lightgrey", lw=0.5)

    plt.savefig("upstream_vs_downstream.pdf")
    fig.set_figheight(1.9)
    fig.set_figwidth(7)
    plt.savefig("upstream_vs_downstream_poster.png", dpi=1200)
    plt.close()
