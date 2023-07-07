from plots_constants import *
from scipy.stats import spearmanr

# Overview of best models w.r.t. R1 and AUROC
with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(2.4)
    chosen_avg = chosen_avg.sort_values(["Sweep"], ascending=[False])
    for s, x, y in zip(chosen_avg["Sweep"].to_list(),
                       chosen_avg["best_test_avg_downstream_r1_mean"].to_list(),
                       chosen_avg["best_test_avg_downstream_auroc_correct_mean"].to_list()):
        marker = id_to_marker[s]
        size = id_to_size[s]
        color = id_to_col[s]
        x_r1 = chosen_r1_avg["best_test_avg_downstream_r1_mean"][chosen_r1_avg["Sweep"] == s]
        y_r1 = chosen_r1_avg["best_test_avg_downstream_auroc_correct_mean"][chosen_r1_avg["Sweep"] == s]
        ax.plot([x, x_r1],
                [y, y_r1],
                c=color,
                zorder=2)

        ax.scatter([x, x_r1],
                   [y, y_r1],
                   marker=marker,
                   c=color,
                   s=size,
                   zorder=3)

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.set_xlabel("Downstream R@1")
    ax.set_ylabel("Downstream R-AUROC")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)
    plt.savefig("overview_tradeoff.pdf")
    fig.set_figheight(1.9)
    plt.savefig("overview_tradeoff_poster.png", dpi=1200)
    plt.close()


# Per model
for resnet in [True, False]:
    with plt.rc_context(bundles.neurips2023()):
        fig = plt.figure()
        fig.set_figheight(7.4)
        for idx, (gp_name, gp) in enumerate(res[res["is_resnet"] == resnet].groupby(by=["Sweep"])):
            ax = fig.add_subplot(4, 3, 1 + idx)
            for s, x, y, color, marker, size in zip(gp["Sweep"].to_list(),
                                                 gp["best_test_avg_downstream_r1"].to_list(),
                                                 gp["best_test_avg_downstream_auroc_correct"].to_list(),
                                                 gp["color"].to_list(),
                                                 gp["marker"].to_list(),
                                                 gp["size"].to_list()):

                ax.scatter(x, y,
                           marker=marker,
                           c=color,
                           s=size,
                           zorder=3)

            rankcorr = spearmanr(gp["best_test_avg_downstream_r1"].to_numpy(),
                                 gp["best_test_avg_downstream_auroc_correct"].to_numpy(), nan_policy="omit")[0]
            ax.text(.02, .98, f'Rank Corr. = {rankcorr:.2f}', ha='left', va='top', transform=ax.transAxes)
            ax.text(.02, 0.02, gp["Sweep"].to_list()[0], color=gp["color"].to_list()[0], ha='left', va='bottom', transform=ax.transAxes)

            ax.set_xlabel("Downstream R@1")
            ax.set_ylabel("Downstream R-AUROC")
            ax.grid(zorder=-1, color="lightgrey", lw=0.5)

        plt.savefig(f"overview_tradeoff_{'resnet' if resnet else 'vit'}.pdf")
        plt.close()
