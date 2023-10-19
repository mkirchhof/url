from plots_constants import *

# Complete Overview plot
with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(2.4)
    for x, y, color, marker, size in zip(res["best_test_avg_downstream_r1"].to_list(),
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

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    ax.set_xlabel("Downstream R@1")
    ax.set_ylabel("Downstream R-AUROC")
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)
    plt.savefig("overview.pdf")
    plt.close()

# Overview of best hyperparameter models
with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(2.4)
    for x, y, color, marker, size in zip(chosen["best_test_avg_downstream_r1"].to_list(),
                                         chosen["best_test_avg_downstream_auroc_correct"].to_list(),
                                         chosen["color"].to_list(),
                                         chosen["marker"].to_list(),
                                         chosen["size"].to_list()):
        ax.scatter(x,
                   y,
                   marker=marker,
                   c=color,
                   s=size,
                   zorder=2)

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.set_xlabel("Downstream R@1")
    ax.set_ylabel("Downstream R-AUROC")
    plt.text(0.615, 0.492, "small = ResNet 50", fontsize=7, zorder=10)
    plt.text(0.615, 0.485, "big = ViT-Medium", fontsize=7, zorder=10)
    ax.grid(zorder=-1, color="lightgrey", lw=0.5)
    plt.savefig("overview_final_hyperparameters.pdf")
    fig.set_figheight(1.9)
    plt.savefig("overview_final_hyperparameters_poster.png", dpi=1200)
    plt.close()
